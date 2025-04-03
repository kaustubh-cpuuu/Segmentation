# Unet 

## Description 
U2net for is used for segmentating salient object 
model is to be trained on inhouse dataset 

## Image Segmentation Archicture for more accurate Image Segmentation.
### This Repository contain code of Unet++ , U2net , U2net with Resnet Backboen , Attention Unet 
```swift 
### 1. Git Clone 
### 2 . Replace the dataset path with your Dataset In train,py
### 3 . Replace with your color code (BGR ) and  Classes Respectively with your RGB Mask.

Check `Rtrain.py` to modify PATH TO TRAINING DATA.

```


## Docker Run
To RUN the model using Docker, follow these steps:

1. **Build the Docker Image**
   Navigate to the directory containing the Dockerfile and run: 

```swift 
docker build -t (image_name) 
```

2. **Run the Docker Container**
After the image is built, you can run the container using 
```swift 
docker run  --shm-size 16G --gpus all -it -d (image_name) python Rtrain.py
```
This command will start the model serving on port 8000.
