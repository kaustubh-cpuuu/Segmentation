FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install albumentations
RUN pip install scipy
RUN pip install Pillow
RUN pip install segmentation-models-pytorch
RUN pip install matplotlib
RUN apt-get update

# Copy the inference script and model
COPY Rtrain.py /workspace/
COPY Rmodel.py /workspace/
COPY Rresnet.py /workspace/
COPY Rutils.py /workspace/
COPY atten_model.py /workspace/

COPY resnet50-19c8e357.pth /workspace/

COPY HBX_new_added_files /workspace/HBX_new_added_files
COPY files /workspace/files

WORKDIR /workspace
