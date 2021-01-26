FROM julia
ARG APCA_API_KEY_ID
ENV APCA-API-KEY-ID=$APCA_API_KEY_ID
ARG APCA_API_SECRET_KEY
ENV APCA-API-SECRET-KEY=$APCA_API_SECRET_KEY
WORKDIR /app
RUN apt-get update && apt-get install -y curl && apt-get install wget -y
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-324.0.0-linux-x86_64.tar.gz
RUN gzip -cd google-cloud-sdk-324.0.0-linux-x86_64.tar.gz|tar -xvf -
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /app/miniconda
ENV PATH $PATH:/app/google-cloud-sdk/bin:/app/miniconda/bin
COPY *.jl /app/
COPY OsRSIConv/ /app/OsRSIConv/
COPY TradeAPI/  /app/TradeAPI/
RUN julia installPkg.jl
ENTRYPOINT ["julia", "run.jl"]
