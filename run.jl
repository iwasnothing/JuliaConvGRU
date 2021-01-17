push!(LOAD_PATH, "ConvGRU/");
using ConvGRU
Base.compilecache(Base.PkgId(ConvGRU))
list=["gbtc","2840.HK","GLD","SQ","BLOK","BLCN"]
using DataFrames
result = DataFrame(symbol=String[],loss=Float32[],time=Float64[])

for sym in list
    tstart = time()
    loss = ConvGRU.mainTest(sym)
    tend=time()
    et=tend-tstart
    push!(result,(sym,loss,et))
end

print(result)
