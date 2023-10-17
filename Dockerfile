FROM ubuntu:20.04

EXPOSE 8189

RUN apt-get update && apt-get install -y curl

CMD ["curl", "--version"]

