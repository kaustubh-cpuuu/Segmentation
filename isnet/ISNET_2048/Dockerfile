FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-glx libglib2.0-dev libopencv-dev


COPY pip_requirements.txt .
RUN pip install -r pip_requirements.txt
RUN pip install tensorboard
RUN pip install comet_ml

WORKDIR /workspace

