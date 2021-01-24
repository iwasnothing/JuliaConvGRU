using Pkg
Pkg.update()


pkgList = ["Sockets","DataFramesMeta","PackageCompiler","CSV","Flux","HTTP","JSON","DataFrames","Plots","TimeSeries","MultivariateStats","Hyperopt","JSON"]
for pkg in pkgList
    Pkg.add(pkg)  
end


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
using JSON


push!(LOAD_PATH, "OsRSIConv/");
using OsRSIConv
push!(LOAD_PATH, "TradeAPI/");
using TradeAPI
