FROM julia
ARG APCA_API_KEY_ID
ENV APCA-API-KEY-ID=$APCA_API_KEY_ID
ARG APCA_API_SECRET_KEY
ENV APCA-API-SECRET-KEY=$APCA_API_SECRET_KEY
WORKDIR /app
RUN apt-get update && apt-get install -y curl
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-324.0.0-linux-x86_64.tar.gz
RUN gzip -cd google-cloud-sdk-324.0.0-linux-x86_64.tar.gz|tar -xvf -
ENV PATH $PATH:/app/google-cloud-sdk/bin
COPY *.jl /app/
COPY OsRSIConv/ /app/OsRSIConv/
COPY TradeAPI/  /app/TradeAPI/
RUN julia installPkg.jl
ENTRYPOINT ["julia", "run.jl"]
