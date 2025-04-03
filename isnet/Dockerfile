FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime


RUN apt-get update
RUN apt-get update
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install matplotlib
RUN pip install comet_ml
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install numpy
RUN pip install albumentations       
RUN pip install scipy
RUN pip install Pillow
RUN pip install timm
RUN pip install tensorboard
RUN pip install plotly
RUN pip install scikit-image
RUN apt-get update

# Copy the inference script and model
COPY __init__.py /workspace/
COPY basics.py /workspace/
COPY data_loader_cache.py /workspace/
COPY hce_metric_main.py /workspace/
COPY isnet.py /workspace/
COPY train_valid_inference_main.py /workspace/


COPY saved_models /workspace/saved_models
# COPY checkpoint /workspace/checkpoint

WORKDIR /workspace


