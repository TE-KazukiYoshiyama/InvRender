FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04


COPY requirement.txt /opt/requirements.txt

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update

RUN apt-get install -y \
  python3 \
  python3-pip \
  libsm6 \
  libxrender1 \
  libxext-dev


RUN pip3 install --upgrade pip
RUN pip3 install -r /opt/requirements.txt