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

push!(LOAD_PATH, "OsRSIConv/");
using DataFrames
using OsRSIConv
push!(LOAD_PATH, "TradeAPI/");
using TradeAPI


list = ["FB","AAPL","AMZN","GOOG","NFLX","SQ","MTCH","AYX","ROKU","TTD"]

result = DataFrame[]

for sym in list
    tstart = time()
    df = OsRSIConv.trainPredict(sym)
    tend=time()
    et=tend-tstart
    push!(result,df)
    if df[1,:accuracy] > 0.55 && df[1,:future] > 0
        TradeAPI.placeOrder(sym)
    end
end

print(vcat(result))
