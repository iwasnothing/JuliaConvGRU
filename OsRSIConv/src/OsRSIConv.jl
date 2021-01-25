module OsRSIConv

using Pkg
Pkg.update()


pkgList = ["Sockets","DataFramesMeta","CSV","Flux","HTTP","JSON","DataFrames","Plots","TimeSeries","Hyperopt","JSON"]
for pkg in pkgList
    Pkg.add(pkg)  
end


using TimeSeries
using DataFrames,DataFramesMeta
using Dates
using Statistics
using Flux
using Flux: @epochs
using Plots
using Hyperopt
using JSON
using HTTP

push!(LOAD_PATH, "TradeAPI/");
using TradeAPI



function toPL(PL_ts)
    function f_PL(values)
        open = values[1]
        close = values[2]
        if close - open > 0
            pl = 1
        else
            pl = 0
        end
        [pl,0]
    end     
    PL = TimeSeries.rename(TimeSeries.map((timestamp, values) -> (timestamp, f_PL(values)),PL_ts)[:Open],Symbol("PL") )
    PL = TimeSeries.map((timestamp, values) -> (timestamp, Int(values)),PL)
end

function toRSI(price_ts,loopback)
    pct = percentchange(price_ts)
    upidx = findall(pct["Close"] .> 0)
    downidx = findall(pct["Close"] .< 0)
    up = map( (timestamp, values) -> (timestamp, if values < 0 ; 0 ; else ;values ;end), pct)
    down = map( (timestamp, values) -> (timestamp, if values < 0 ; abs(values) ; else ;0 ;end), pct)
    up_roll = moving(mean, up, loopback)
    down_roll = moving(mean, down, loopback)
    function f_rsi(values)
        x = values[1]
        y = values[2]
        rsi = 100 - (100 / (1 + x/y) )
        [rsi,0]
    end
    updown = TimeSeries.rename(TimeSeries.merge(up_roll,down_roll), [:up,:down])
    rsi_ts = TimeSeries.rename(TimeSeries.map((timestamp, values) -> (timestamp, f_rsi(values)), updown)[:up],Symbol("RSI-",loopback))
    return rsi_ts
end

function toORSI(price_ts,day0,day1)
    #pct = percentchange(price_ts)
    result_rsi = price_ts
    result_rsima = price_ts
    result_orsi = price_ts
    for i in day0:day1
        rsi_ts = toRSI(price_ts,i)
        result_rsi = merge(result_rsi,rsi_ts,method=:inner)
        for j in day0:day1
            rsi_ma = TimeSeries.rename(moving(mean,rsi_ts,j),Symbol("RSIMA-",i,"-",j))
            orsi = TimeSeries.rename(rsi_ts .- rsi_ma , Symbol("ORSI-",i,"-",j) )
            result_orsi = merge(result_orsi,orsi,method=:inner)
        end
    end
    #println(colnames(result_orsi))
    return result_orsi
end

function toMatrixORSI(data,featureIdx,targetIdx)
    println(colnames(data)[1:5])
    #featureIdx=3
    data = values(data)
    X = data[:,featureIdx:end]
    #b = maximum(X)
    #a = minimum(X)
    #println(a,"-",b)
    #X = ( b .- X ) ./ (b-a+1)
    print(size(X))
    N = size(X)[1]
    k = Int(sqrt(size(X)[2]))
    println(k)
    println(X[1,1:4])
    println(X[2,1:4])
    println(X[2,k+1:k+4])
    X = transpose(X)
    X = reshape(X,(k,k,N))
    println(X[1:4,1,1])
    println(X[1:4,1,2])
    println(X[1:4,2,2])
    #targetIdx=1
    Y = data[:,targetIdx]
    return (X,Y)
end

function toReturn(price_ts)
    pct = TimeSeries.rename(percentchange(price_ts),[:return])
    pct = moving(mean,pct,5)
    return TimeSeries.map((timestamp, values) -> (timestamp, 100*values), pct)
end   

function preprocessing(etf,day0,day1,mlclass)
    PL = select(etf,[:Open,:Close,:ondate])
    price = select(etf,[:Close,:ondate])
    PL_ts = TimeArray(PL,timestamp=:ondate)
    PL_ts = toPL(PL_ts)
    pct = TimeSeries.rename(percentchange(TimeArray(price,timestamp=:ondate)),[:return])
    price_ts = TimeArray(price,timestamp=:ondate)
    pct = toReturn(price_ts)
    #day0=5
    #day1=30
    result_orsi = toORSI(price_ts,day0,day1)
    tgt = merge(PL_ts,pct,method=:inner)
    data = merge(tgt,result_orsi,method=:inner)
    featureIdx = 4
    if mlclass == :reg
        tgt = 2
    else
        tgt = 1
    end
        
    (X,Y) = toMatrixORSI(data,featureIdx,tgt)
end

function timeseriesDataset(X,Y,seqlen,mlclass)
    xtrain = Array{Float32,3}[]
    ytrain = Array{Float32,1}[]
    len = length(Y)
    for i in 1:len-seqlen-1
        mx = X[:,:,i:i+seqlen-1]
        xtrain = vcat(xtrain,[mx])
        if mlclass == :reg
            my = Y[i+seqlen]
        else
            my = Flux.onehot(Y[i+seqlen], [0,1])
        end
        ytrain = vcat(ytrain,[my])
    end
    xcurrent = [X[:,:,end-seqlen+1:end]]
    return (xtrain,ytrain,xcurrent)
end
    

function build_model(Nh,seqlen, delta)
    a = floor(Int8,Nh)
    sz = ceil(Int8,delta/8)
    
    println(delta,"-",sz)
    return Chain(
    x -> Flux.unsqueeze(x,4),
    # First convolution, operating upon a 28x28 image
    Conv((2, 2), seqlen=>a, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((2, 2), a=>Nh, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((2, 2), Nh=>Nh, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    Flux.flatten,
    #Dropout(0.1),
    Dense(sz*sz*Nh, 2),
    softmax
    )
end

function build_reg_model(Nh,seqlen, delta)
    a = floor(Int8,Nh)
    sz = ceil(Int8,delta/8)
    
    println(delta,"-",sz)
    return Chain(
    x -> Flux.unsqueeze(x,4),
    # First convolution, operating upon a 28x28 image
    Conv((2, 2), seqlen=>a, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((2, 2), a=>Nh, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((2, 2), Nh=>Nh, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    Flux.flatten,
    Dropout(0.1),
    (x->transpose(x)),
    GRU(1,Nh),
    GRU(Nh,Nh),
    (x -> x[:,end]),
    Dense(Nh, 1),
    (x -> x[1])
    )
end

function accuracy(m,xtest,ytest)
    prediction = m.(xtest)
    len = length(ytest)
    m = zeros(2,2)
    for i in 1:len
        yh = Flux.onecold(prediction[i],[0,1])
        y = Flux.onecold(ytest[i],[0,1])
        a = Int(yh[1])+1
        b = y+1
        m[a,b] += 1
    end
    m = m ./ sum(m)
    return m
end

function reg_accuracy(m,xtest,ytest)
    prediction = m.(xtest)
    len = length(ytest)
    m = zeros(2,2)
    for i in 1:len
        yh = prediction[i] > 0 ? 1 : 0
        y = ytest[i] > 0 ? 1 : 0
        a = Int(yh[1])+1
        b = y+1
        m[a,b] += 1
    end
    m = m ./ sum(m)
    return m
end

function plot_eval(m,xtest,ytest)
    prediction = m.(xtest)
    len = length(ytest)
    #println(len)
    pred = Float32[]
    actual = Float32[]
    total = 0
    for i in 1:len
        yh = Flux.onecold(prediction[i],[0,1])
        y = Flux.onecold(ytest[i],[0,1])
        append!(pred,yh)
        append!(actual,y)
    end
    plot(1:len,pred,size = (1000, 700),color="red",tickfontsize=28,guidefontsize=28,legendfontsize=28)
    plot!(1:len,actual,size = (1000, 700),color="blue",tickfontsize=28,guidefontsize=28,legendfontsize=28)
end

function myTrain(etf,seqlen,Nh,lr,mm,day0,day1, mlclass,i)
    #seqlen=7
    #mlclass = :clf
    (X,Y) = preprocessing(etf,day0,day1,mlclass)
    (xtrain,ytrain,xcurrent) = timeseriesDataset(X,Y,seqlen,mlclass)
    thd = 150
    xtest,ytest = xtrain[end-thd+1:end], ytrain[end-thd+1:end]
    xtrain,ytrain = xtrain[1:end-thd], ytrain[1:end-thd]
    batchsize = 20
    train_loader = Flux.Data.DataLoader(xtrain,ytrain, batchsize=batchsize,shuffle=false)
    #Nh=10
    delta = day1 - day0 + 1
    if mlclass == :reg
        m = build_reg_model(Nh,seqlen,delta)
    else
        m = build_model(Nh,seqlen,delta)
    end
    function loss(x, y)    
        yh = m.(x)
        #println(yh)
        #println(y)
        #println(Flux.Losses.logitcrossentropy(yh[1], y[1]))
        return sum(Flux.Losses.logitcrossentropy.(yh, y))
    end
    function acc_loss(x,y)
        (acc,y0,y1,y2) = accuracy(m,x,y)
        return (1-acc-y0+y1)*abs(y2-0.5)
    end
    function mse_loss(x,y)
        yh = m.(x)
        e = Flux.mae(yh,y)
        return e
    end
    evalcb() = @show loss(xtest,ytest)
    num_epoch = 25
    #lr = 0.01
    if mlclass == :reg
        @epochs num_epoch Flux.train!(mse_loss,Flux.params(m),train_loader,RMSProp(lr,mm))
        confmx = reg_accuracy(m,xtest,ytest)
        prediction = m.(xtest)
        println(size(prediction))
        mseloss = mse_loss(xtest,ytest)
        println("mse=",mseloss)
        objcost = mseloss-1*(confmx[1,1]+confmx[2,2]-confmx[1,2]-confmx[2,1])
        #p = map(x -> x[end],prediction)
        #y = map(x -> x[end],ytest)
        #lp = getindex(size(prediction),1)
        #ly = getindex(size(y),1)
        plot(1:length(prediction),prediction,size = (1000, 700),color="red",tickfontsize=28,guidefontsize=28,legendfontsize=28)
        plot!(1:length(ytest),ytest,size = (1000, 700),color="blue",tickfontsize=28,guidefontsize=28,legendfontsize=28)
        png("plot-"*string(i)*".png")
        println("current size:", size(xcurrent[1]))
        future = m.(xcurrent)
        future = future[1]
        println("future:",future)
    else
        @epochs num_epoch Flux.train!(loss,Flux.params(m),train_loader,RMSProp(lr,mm))
        confmx = accuracy(m,xtest,ytest)
        objcost = -1*(confmx[1,1]+confmx[2,2]-confmx[1,2]-confmx[2,1])
        println("current size:", size(xcurrent[1]))
        future = m.(xcurrent)
        future = future[1]
        println("future:",future)
    end
    println("accuracy:",(confmx[1,1]+confmx[2,2]))
    println(confmx[1,:])
    println(confmx[2,:])
    
    return (m,objcost,xtest,ytest,confmx,future)
end
function myObjective(etf,seqlen,Nh,lr,mm,day0,day1,mlclass,i)
    (m,objcost,xtest,ytest,confmx,future) = myTrain(etf,seqlen,Nh,lr,mm,day0,day1,mlclass,i)
    Flux.reset!(m)
    return objcost
end

function hyperTune(sym::String)
    etf = TradeAPI.wgetStock(sym,3)
    ho = @hyperopt for i=20,
                sampler = RandomSampler(), # This is default if none provided
                seqlen = StepRange(3, 5, 20),
                Nh = StepRange(5,3, 20),
                delta = StepRange(10,5, 25),
                #lr =  LinRange(1e-5,9e-5,10),
                lr =  exp10.(LinRange(-4,-3,10)),
                mm =  LinRange(0.75,0.95,5),
                day0 = StepRange(5,3, 10)
        println(i,"-",seqlen,"-",Nh,"-",lr,"-",mm,"-",day0,"-",delta)
    @show myObjective(etf,seqlen,Nh,lr,mm,day0,day0+delta,:reg,i)
    end

    best_params, min_f = ho.minimizer, ho.minimum

    seqlen=best_params[1]
    Nh=best_params[2]
    delta=best_params[3]
    lr=best_params[4]
    mm=best_params[5]
    day0=best_params[6]
    (m,acc,xtest,ytest,confmx,future) = myTrain(etf,seqlen,Nh,lr,mm,day0,day0+delta,:reg,21)
end

function trainPredict(sym::String)
    #20-13-20-0.000774263682681127-0.8-5-15
    seqlen=13
    Nh=20
    delta=15
    lr=0.000774
    mm=0.8
    day0=5
    println(seqlen,"-",Nh,"-",lr,"-",mm,"-",day0,"-",delta)
    df = DataFrame(symbol = String[], accuracy = Float32[], future = Float32[])
    etf = TradeAPI.wgetStock(sym,3)
    (m,cost,xtest,ytest,confmx,future) = myTrain(etf,seqlen,Nh,lr,mm,day0,day0+delta,:reg,sym)
    acc = (confmx[1,1]+confmx[2,2])
    push!(df,[sym,acc,future])
    println(df)
    return df
end

export trainPredict
export hyperTune
#precompile(trainPredict, (String,)) 
#precompile(hyperTune, (String,))
end # module
