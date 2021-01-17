module ConvGRU

using Pkg

ENV["PYTHON"] = "/Users/kahingleung/PycharmProjects/mylightning/venv/bin/python3.8"
Pkg.build("PyCall")
using PyCall
using Flux
using Flux: @epochs
using HTTP
using JSON
using Pkg
using DataFrames,DataFramesMeta
using Dates
using Statistics
using Statistics: mean
using Printf
using Plots
using Base
using CSV
using Hyperopt
using TimeSeries
using MultivariateStats



function getFX(fromYear,toYear)
    #url_query = "https://api.exchangeratesapi.io/history?start_at="*fromYear*"&end_at="*toYear*"&symbols=PHP,THB,IDR,GBP&base=USD"
    url_query = "https://api.exchangeratesapi.io/history?start_at="*fromYear*"&end_at="*toYear*"&base=USD"
    raw_response = HTTP.request("GET", url_query)
    rsp = JSON.parse(String(raw_response.body))
    rates = get(rsp,"rates",0)
    #df1 = DataFrame(date=String[],PHP=Float64[],THB=Float64[],IDR=Float64[],GBP=Float64[])
    df = DataFrame(date=String[], CUR=String[], rate=Float32[])
    for (k,v) in rates
        #print(k,get(v,"EUR",0) ,get(v,"JPY",0))
        #d = Dict("date" => k, "PHP" => get(v,"PHP",0), "THB" => get(v,"THB",0), "IDR" => get(v,"IDR",0), "GBP" => get(v,"GBP",0))
        for cur in keys(v)
            r = get(v,cur,0)
            row = DataFrame(date = [k], CUR=[cur], rate=[r])
            df = vcat(df,row)
        end
    end
    transform!(df, :date => ( x -> Date.(x, Dates.DateFormat("yyyy-mm-dd")) ) => :ondate)
    return df
end
function getBTC(from,to)
    headers = ["apikey" => "101c5169-770f-4add-8be0-6131419b7208"]
    url_query = "http://api.bitdataset.com/v1/ohlcv/history/BITFINEX:BTCUSD?period=D1&start="*from*"&end="*to*"&limit=1000"
    raw_response = HTTP.request("GET", url_query, headers)
    rsp = JSON.parse(String(raw_response.body))
    df2 = DataFrame(high = Float32[],low = Float32[],open = Float32[],close = Float32[],volume = Float32[], time = String[])
    for d in rsp
        push!(df2,d)
    end
    transform!(df2, :time => ( x -> split.(x, "T") ) => :dateary)
    transform!(df2, :dateary => ( x -> getindex.(x,1 ) ) => :datestr)
    transform!(df2, :datestr => ( x -> Date.(x, Dates.DateFormat("yyyy-mm-dd")) ) => :ondate)
    return df2
end
function loadData(from,to)
    d2 = getFX(from,to)
    d1 = getBTC(from,to)
    
    df3 = innerjoin(select(d1,[:open,:close,:high,:low,:volume,:ondate]), d2, on = :ondate)
    return df3
end

function loadData()
    y0 = Dates.format(today(), "yyyy-mm-dd")
    y1 = Dates.format(today()-Year(1), "yyyy-mm-dd")
    y2 = Dates.format(today()-Year(2), "yyyy-mm-dd")
    y3 = Dates.format(today()-Year(3), "yyyy-mm-dd")

    infile="/Users/kahingleung/Downloads/gold-julia/btc.csv"
    if isfile(infile)
        rawbtc = CSV.File(infile)
        rawbtc = DataFrame(rawbtc)
        for c in ["high","low","open","close","volume"]
            transform!(rawbtc,c => (x -> Float32.(x) )=> c)
        end
    else
        btc1 = getBTC(y1,y0)
        btc2 = getBTC(y2,y1)
        btc3 = getBTC(y3,y2)
        rawbtc = vcat(btc1,btc2,btc3)
        CSV.write(infile, rawbtc)
    end
    rawbtc

    infile="/Users/kahingleung/Downloads/gold-julia/fx.csv"
    if isfile(infile)
        rawdata = CSV.File(infile)
        rawdata = DataFrame(rawdata)
        for c in ["rate"]
            transform!(rawdata,c => (x -> Float32.(x) )=> c)
        end
    else
        df1 = getFX(y1,y0)
        df2 = getFX(y2,y1)
        df3 = getFX(y3,y2)
        rawdata = vcat(df1,df2,df3)
        CSV.write(infile, rawdata)
    end
    rawdata

    rawdata = select(rawdata,[:ondate,:CUR,:rate])
    rawdata = combine(groupby(rawdata, [:ondate,:CUR]),:rate=>mean)

    df = DataFrame()
    df.ondate = unique(rawdata.ondate)
    datecol = df
    for c in unique(rawdata.CUR)
        df1 = @linq rawdata |> where(:CUR .== c) |> select(:ondate,:rate_mean)
        coldata = leftjoin(datecol,df1,on=:ondate)
        df[:,c] = coldata[:,:rate_mean]
    end
    df = sort(df,order(:ondate))

    dropmissing!(df)

    df = innerjoin(select(rawbtc,[:open,:close,:high,:low,:volume,:ondate]), df, on = :ondate)

    return df
end



function loadStock(sym::String)
    #sym = "gbtc"
    yf = pyimport("yfinance")
    ticker = yf.Ticker(sym)
    etf = ticker.history(period="3y")

    function pd_to_df(df_pd)
        df= DataFrame()
        for col in df_pd.columns
            df[!, col] = getproperty(df_pd, col).values
        end
        df[!,:Date] = collect(df_pd[:index])
        return df
    end
    etf = pd_to_df(etf)

    transform!(etf, :Date => ( x -> Date.(Dates.year.(x),Dates.month.(x),Dates.day.(x)) ) => :ondate)
    for c in ["High","Low","Open","Close","Stock Splits"]
        transform!(etf,c => (x -> Float32.(x) )=> c)
    end

    return select(etf,["Close","ondate"])
end

function preprocessing(df,etf)
    df3 = innerjoin(df, etf, on = :ondate)
    sort!(df3, [:ondate])


    println(names(df3))

    ts = TimeArray(df3,timestamp=:ondate)

    pct = percentchange(ts)
    pct = moving(mean, pct, 5)

    df = DataFrame(pct)

    target = ["Close"]
    features = [c for c in names(df) if c != "timestamp" && c != "Close"]
    #features = ["AUD","GBP","open","close","high","low","volume"]


    for col in vcat(features, target)
        transform!(df,[col] => (x -> map(a->Float32(100*a),x)) => [col]) 
    end


    dropmissing!(df)
    return select(df,vcat(features, target))
end

function timeseriesDataset(df,seqlen,features,target)
    xtrain = Matrix[]
    ytrain = Float32[]
    #seqlen=7
    f = getindex(size(features),1)
    #println(f)
    len=getindex(size(df),1)
    mx = transpose(convert(Matrix,df[:,features]))
    #println(size(mx))
    M = MultivariateStats.fit(PCA, mx; maxoutdim=3)
    Yte = MultivariateStats.transform(M, mx)
    #println(size(Yte))
    #Yte = mx
    for i in 1:len-seqlen-1
        #mx = select(df[i:i+seqlen-1,:],features)
        #mx = reshape(convert(Matrix,mx),(f,seqlen))
        #print(size(mx))
        mx = Yte[:,i:i+seqlen-1]
        #println(size(mx))
        xtrain = vcat(xtrain,[mx])

        #my = df[i+1:i+seqlen,target]
        my = df[i+seqlen,target]
        #my = getindex(convert(Vector,my),1)
        #println(size(my))
        ytrain = vcat(ytrain,[my])
    end
    return (xtrain,ytrain)
end





function build_model(Nh)
    a = floor(Int8,Nh)
    return Chain(
    x -> Flux.unsqueeze(Flux.unsqueeze(x,3),4),
    # First convolution, operating upon a 28x28 image
    Conv((2, 2), 1=>a, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((2, 2), a=>Nh, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    #Conv((2, 2), Nh=>Nh, pad=(1,1), relu),
    #MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    Flux.flatten,
    Dropout(0.1),
    (x->transpose(x)),
    GRU(1,Nh),
    GRU(Nh,Nh),
    (x -> x[:,end]),
    Dense(Nh, 1),
    (x -> x[1]))
end

function build_lstm()
    #Nt = seqlen       # time steps
    Nin,Nout = 3,1 # input size, output size
    #Nh = 5        # hidden dim
    #num_layers = 2
    layers = []
    layers = vcat(layers,[LSTM(Nin,Nh),Dropout(dr)])
    for i in 1:num_layers-1
        layers = vcat(layers,[LSTM(Nh,Nh),Dropout(dr)])
    end
    #layers = vcat(layers,[Dropout(dr)])
    linear1 = Chain(Dense(Nh,10),Dense(10,1))
    #lstm = Chain(Dense(Nin,Nh),Dense(Nh,1))
    #lstm(x) = linear1(foldl((x, m) -> m(x), layers, init = x)[:,end])
    lstm = Chain(LSTM(Nin,Nh),Dropout(dr),Dense(Nh,10),Dense(10,Nout)) # simple lstm
    #lstm(x) = linear1(foldl((x, m) -> m(x), layers, init = x))
    return lstm
end



function myTrain(df,seqlen,Nh,lr,num_epoch,suffix)
    features = names(df)[1:end-2]
    target = names(df)[end]
    (xtrain,ytrain) = timeseriesDataset(df,seqlen,features,target)
    #println(size(xtrain[1]))
    thd = 250
    xtest,ytest = xtrain[end-thd+1:end], ytrain[end-thd+1:end]
    xtrain,ytrain = xtrain[1:end-thd], ytrain[1:end-thd]
    batchsize = 20
    train_loader = Flux.Data.DataLoader(xtrain,ytrain, batchsize=batchsize,shuffle=false)
    m = build_model(Nh)
    function mse_loss(x,y)
        yh = m.(x)
        e = Flux.mae(yh,y)
        return e
    end
    evalcb() = @show mse_loss(xtest,ytest)

    throttled_cb = @show mse_loss(xtest,ytest)
    
    
    @epochs num_epoch Flux.train!(mse_loss,Flux.params(m),train_loader,RMSProp(lr),cb = Flux.throttle(evalcb, 2))
    
    prediction = m.(xtest)
    println(size(prediction))
    finalLoss = mse_loss(xtest,ytest)
    p = map(x -> x[end],prediction)
    y = map(x -> x[end],ytest)
    lp = getindex(size(prediction),1)
    ly = getindex(size(y),1)
    plot(1:lp,p,size = (1000, 700),color="red",tickfontsize=28,guidefontsize=28,legendfontsize=28)
    plot!(1:ly,y,size = (1000, 700),color="blue",tickfontsize=28,guidefontsize=28,legendfontsize=28)
    dir="/Users/kahingleung/Downloads/gold-julia/"
    png("/Users/kahingleung/Downloads/gold-julia/plot-"*suffix*".png")
    return (m,ytest,prediction,finalLoss)
end

function myObjective(df,seqlen,Nh,lr,suffix)
    (lstm,ytest,prediction,finalLoss) = myTrain(df,seqlen,Nh,lr,30,suffix)
    Flux.reset!(lstm)
    return finalLoss
end

function mainTest(sym::String)
    #sym = "gbtc"
    fx = loadData()
    etf = loadStock(sym)
    df = preprocessing(fx,etf)
    ho = @hyperopt for i=20,
                sampler = RandomSampler(), # This is default if none provided
                seqlen = StepRange(10, 5, 30),
                Nh = StepRange(2,2, 8),
                lr =  exp10.(LinRange(-3,-2,10))
    println(i,seqlen,Nh,lr)
    @show myObjective(df,seqlen,Nh,lr,sym*"-"*string(i))
    end

    best_params, min_f = ho.minimizer, ho.minimum

    seqlen=best_params[1]
    Nh=best_params[2]
    lr=best_params[3]
    (m,ytest,prediction,finalLoss) = myTrain(df,seqlen,Nh,lr,100,sym*"-21")
    println(finalLoss)


    sum = 0
    N = length(ytest)
    for (x,y) in zip(ytest,prediction)
        s = x*y
        if s > 0
            sum += 1
        end
    end
    println(sum/N)

    return sum/N
end


export mainTest
precompile(mainTest, (String,)) 
precompile(myTrain, (DataFrame,Int,Int,Float32,Int,String) )



end # module
