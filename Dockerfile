FROM julia
WORKDIR /app
COPY *.jl .
COPY ConvGRU/ /app/ConvGRU/
RUN find /app
RUN julia /app/compile.jl
ENTRYPOINT ["julia", "run.jl"]
