FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install numpy
RUN pip install albumentations
RUN pip install scipy
RUN pip install Pillow
RUN pip install timm
RUN apt-get update

# Copy the inference script and model
COPY __init__.py /workspace/
COPY basics.py /workspace/
COPY data_loader_cache.py /workspace/
COPY hce_metric_main.py /workspace/
COPY Inference.py /workspace/
COPY isnet.py /workspace/
COPY isnet_resnet50.py /workspace/
COPY isnet_DenseNet_121.py /workspace/
COPY isnet_vision_trnsformer_resnet50.py /workspace/
COPY train_valid_inference_main.py /workspace/


#COPY 1000_images /workspace/1000_images
COPY 216_images /workspace/216_images
COPY saved_models /workspace/saved_models

WORKDIR /workspace
