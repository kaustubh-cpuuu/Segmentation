import os
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from Atrain import get_colormap
from Amodel import attention_unet
from Autils import seeding, create_dir
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Load and split the dataset """
def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "Training22", "Training22", "*")))[:15]
    train_y = sorted(glob(os.path.join(path, "Training22", "Categories1", "*")))[:15]

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)

    valid_x = sorted(glob(os.path.join(path, "Validation", "Training22", "*")))[:3]
    valid_y = sorted(glob(os.path.join(path, "Validation", "Categories1", "*")))[:3]

    return (train_x, train_y), (valid_x, valid_y), (test_x[:100], test_y[:100])

def grayscale_to_rgb(mask, classes, colormap):
    h, w = mask.shape[:2]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directory for storing files """
    create_dir("results/rgb")

    """ Hyperparameters """
    IMG_H = 512
    IMG_W = 512

    path = "Data"
    model_path = os.path.join("files", "model.h5")

    CLASSES, COLORMAP = get_colormap()
    NUM_CLASSES = len(CLASSES)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attention_unet(num_classes=NUM_CLASSES)
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Load the dataset """
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(path)
    # print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}")
    # print("")

    test_x=["dataaaa/Training/Training/image1.png"]
    test_y=["dataaaa/Training/Training/image1.png"]

    """ Evaluation and Prediction """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
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

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_COLOR)
        mask_x = mask
        mask = cv2.resize(mask, (IMG_W, IMG_H))

        onehot_mask = []
        for color in COLORMAP:
            cmap = np.all(np.equal(mask, color), axis=-1)
            onehot_mask.append(cmap)
        onehot_mask = np.stack(onehot_mask, axis=-1)
        onehot_mask = np.argmax(onehot_mask, axis=-1)
        onehot_mask = onehot_mask.astype(np.int32)

        """ Prediction """
        with torch.no_grad():
            pred = model(image)
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = pred[0].astype(np.float32)

        """ Saving the prediction """
        save_image_path = f"results/rgb/{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        """ Flatten the array """
        onehot_mask = onehot_mask.flatten()
        pred = pred.flatten()

        labels = [i for i in range(NUM_CLASSES)]

        """ Calculating the metrics values """
        f1_value = f1_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)

        SCORE.append([f1_value, jac_value])

    """ Metrics values """
    score = np.array(SCORE)
    score = np.mean(score, axis=0)

    f = open("files/score_rgb.csv", "w")
    f.write("Class,F1,Jaccard\n")

    l = ["Class", "F1", "Jaccard"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    print("-"*35)

    for i in range(0, score.shape[1], 1):
        class_name = CLASSES[i]
        f1 = score[0, i]
        jac = score[1, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    print("-"*35)
    class_mean = np.mean(score, axis=-1)
    class_name = "Mean"
    f1 = class_mean[0]
    jac = class_mean[1]
    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    f.close()

