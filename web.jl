
push!(LOAD_PATH, "OsRSIConv/");
using OsRSIConv
Base.compilecache(Base.PkgId(OsRSIConv))

using HTTP
using Sockets
using DataFrams

function testResponse(request::HTTP.Request)
   @show request
   @show request.method
   @show HTTP.header(request, "Content-Type")
   @show HTTP.payload(request)
   sym="GLD"
   df = OsRSIConv.trainPredict(sym)
   try
       return HTTP.Response("Hello")
   catch e
       return HTTP.Response(404, "Error: $e")
   end
end

const TEST_ROUTER = HTTP.Router()
HTTP.@register(TEST_ROUTER, "GET", "/", testResponse)
host="0.0.0.0"
HTTP.serve(TEST_ROUTER, ip"0.0.0.0", UInt16(8080) )