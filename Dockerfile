FROM horovod/horovod:sha-a1f17d8

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y software-properties-common python3-opencv

COPY requirements.txt /workspace/

RUN pip install -r /workspace/requirements.txt
