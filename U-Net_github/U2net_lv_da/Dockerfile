FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /workspace

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
RUN pip install Pillow
RUN pip install pandas
RUN pip install numpy
RUN pip install tensorboard
RUN apt-get update

COPY train.py /workspace/


COPY 1000_images /workspace/1000_images
COPY data /workspace/data
COPY logs /workspace/logs
COPY networks /workspace/networks
COPY options /workspace/options
COPY prev_checkpoints /workspace/prev_checkpoints
COPY results /workspace/results
COPY utils /workspace/utils


WORKDIR /workspace

