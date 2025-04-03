# IS-Net 

## Description 
IS-Net is for the  Binary segmentation

## Binary Image Segmentation Archicture for more accurate Image Segmentation.
### This Repository contain code of IS-Net with  Backbones  and  Attention Mechanism
```swift 
1.  Git Clone 
2 . Replace the dataset path with your Dataset In train_valid_inference_main.pyC c
3 . Parameters value to be changed from train_valid_inference_main.py

Check `[train_valid_inference_main.py]` to modify PATH TO TRAINING DATA.
```

## Docker Run
To RUN the model using Docker, follow these steps:

1. **Build the Docker Image**
   Navigate to the directory containing the Dockerfile and run: 

```swift 
docker build -t (image_name) .
```

2. **Run the Docker Container**
After the image is built, you can run the container using 
```swift 
docker run  --shm-size 16G --gpus all -it -d (image_name) python train_valid_inference_main.py
```

# Testing/Inference

- Put input images in `Demo_datset / your dataset` folder
- Run `Inference.py` for inference.
- Output will be saved in `Demo Dataset / your dataset results `