module TradeAPI
using Pkg
Pkg.update()


pkgList = ["Sockets","DataFramesMeta","CSV","Flux","HTTP","JSON","DataFrames","Plots","TimeSeries","Hyperopt","JSON"]
for pkg in pkgList
    Pkg.add(pkg)  
end

using HTTP
using JSON
using DataFrames
using Dates
using TimeSeries

function wgetStock(sym::String,period::Int)
    #println("id",ENV["APCA-API-KEY-ID"])
    #println("key",ENV["APCA-API-SECRET-KEY"])
    fromYear = Dates.format(today()-Year(period), "yyyy-mm-dd")
    headers = ["APCA-API-KEY-ID" => ENV["APCA-API-KEY-ID"], "APCA-API-SECRET-KEY" => ENV["APCA-API-SECRET-KEY"] ]
    url_query = "https://data.alpaca.markets/v1/bars/day?symbols="*sym*"&after="*fromYear*"T00:00:00&limit=1000"
    raw_response = HTTP.request("GET", url_query, headers)
    rsp = JSON.parse(String(raw_response.body))
    df2 = DataFrame(t = Int[],h = Float32[],l = Float32[],o = Float32[],c = Float32[],v = Float32[] )
    for (sym,v) in rsp
        for q in v
            push!(df2,q)
        end
    end
    transform!(df2, :t => ( x -> Dates.unix2datetime.(x) ) => :ondate)
    DataFrames.rename!(df2,:h => :High)
    DataFrames.rename!(df2,:l => :Low)
    DataFrames.rename!(df2,:o => :Open)
    DataFrames.rename!(df2,:c => :Close)
    DataFrames.rename!(df2,:v => :Volume)
    sort!(df2,order(:ondate))
    return select(df2,[:Open,:Close,:High,:Low,:Volume,:ondate])
end

function placeOrder(sym::String)
    headers = ["APCA-API-KEY-ID" => ENV["APCA-API-KEY-ID"], "APCA-API-SECRET-KEY" => ENV["APCA-API-SECRET-KEY"] ]
    url_query = "https://data.alpaca.markets//v1/last_quote/stocks/"*sym
    raw_response = HTTP.request("GET", url_query, headers)
    rsp = JSON.parse(String(raw_response.body))
    @show rsp
    @show rsp["last"]["askprice"]
    p1 = rsp["last"]["askprice"]
    spread = 0.1
    @show p1*(1-spread)
    url_query = "https://paper-api.alpaca.markets/v2/orders"
    params =  Dict(
        "symbol"        => sym,
        "qty"           => 10,
        "side"          => "buy",
        "type"          => "market",
        "time_in_force" => "day",
        "order_class"   => "bracket",
        "take_profit"   => Dict("limit_price" => p1*(1+spread)),
        "stop_loss"     => Dict("stop_price" => p1*(1-spread) , "limit_price" => p1*(1-spread)*0.99)
        
       )
    raw_response = HTTP.request("POST", url_query, headers,JSON.json(params))
    rsp = JSON.parse(String(raw_response.body))
    @show rsp

end

export wgetStock
export placeOrder

end # module
