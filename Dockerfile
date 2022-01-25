FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    cmake software-properties-common wget python3-pip python3-opencv vim

COPY requirements.txt /workspace/

## NCCL implements multi-GPU and multi-node communication primitives; required by horovod
#ENV HOROVOD_GPU_OPERATIONS=NCCL

RUN pip3 install --upgrade pip && pip3 install -r /workspace/requirements.txt


###########
# HOROVOD #
###########

# install nvidia NCCL (https://developer.nvidia.com/nccl/nccl-download)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update & apt install libnccl2=2.11.4-1+cuda10.2 libnccl-dev=2.11.4-1+cuda10.2

## install openmpi
#RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz && \
#    tar -xf openmpi-4.1.2.tar.gz && \
#    cd openmpi-4.1.2 && \
#    ./configure --prefix=/usr/local \
#                # cuda support
#                --with-cuda \
#                # c++ bindings needed by horovod
#                --enable-mpi-cxx && \
#    make all install
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/openmpi

# install horovod
RUN #HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 pip3 install --no-cache-dir horovod
RUN HOROVOD_GPU_OPERATIONS=NCCL pip3 install --no-cache-dir horovod
