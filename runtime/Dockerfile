FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=80
ENV JWT_SECRET_KEY=""
ENV TRT_CONCURRENT=1

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++
RUN git lfs install
RUN git clone --recursive https://github.com/viewuz/CosyVoice.git

# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN cd CosyVoice && pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir

RUN cd CosyVoice/runtime/python/fastapi && python3 server.py --port ${PORT} --jwt-secret-key ${JWT_SECRET_KEY} --trt-concurrent ${TRT_CONCURRENT}  && sleep infinity


