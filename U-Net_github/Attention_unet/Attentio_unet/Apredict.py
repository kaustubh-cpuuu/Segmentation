import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch
from train import load_dataset, get_colormap
from Amodel import attention_unet
from Autils import seeding, create_dir

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP
global SPECIFIC_CLASS

def grayscale_to_rgb(mask, classes, colormap):
    h, w = mask.shape[:2]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])

    output = np.reshape(output, (h, w, 3))
    return output


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directory for storing files """
    create_dir("pred/cat")
    create_dir("pred/mask")

    """ Hyperparameters """
    IMG_H = 512
    IMG_W = 512

    path = "/media/nikhil/New Volume/ML_DATASET/instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing"
    model_path = os.path.join("files", "model.h5")

    CLASSES, COLORMAP = get_colormap(path)
    NUM_CLASSES = len(CLASSES)


    SPECIFIC_CLASS = "Face"


    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attention_unet(num_classes=NUM_CLASSES)
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Load the dataset """
    images = glob("/media/nikhil/New Volume/ML_DATASET/instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/Testing/Training1/*.jpg")
    print(f"Training1: {len(images)}")

    """ Prediction """
    for x in tqdm(images, total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image_x = image
        image = cv2.resize(image, (IMG_W, IMG_H))
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Prediction """
        with torch.no_grad():
            pred = model(image)
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = pred[0].astype(np.float32)

        """ Specific Class """
        if SPECIFIC_CLASS:
            class_index = CLASSES.index(SPECIFIC_CLASS)
            pred = np.where(pred == class_index, 255, 0)
            pred = np.expand_dims(pred, axis=-1)
            pred = pred.astype(np.float32)


        """ Saving the prediction """
        h, w, _ = image_x.shape
        line = np.ones((h, 10, 3)) * 255

        pred = cv2.resize(pred, (image_x.shape[1], image_x.shape[0]))
        pred = np.expand_dims(pred, axis=-1)

        if SPECIFIC_CLASS:
            pred = np.concatenate([pred, pred, pred], axis=-1)
        else:
            pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

        cat_images = np.concatenate([image_x, line, pred], axis=1)
        cv2.imwrite(f"pred/cat/{name}.png", cat_images)
        cv2.imwrite(f"pred/mask/{name}.png", pred)
