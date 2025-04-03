import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch
from Rtrain import get_colormap
from Rmodel import build_u2net
from Rutils import seeding, create_dir
from sklearn.model_selection import KFold


""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP


"""K Fold Cross Validation1"""
# def load_dataset(path, split=0.1, n_splits=5, current_fold=0):
#     train_x = sorted(glob(os.path.join(path, "Categories", "Categories", "*")))[:30]
#     train_y = sorted(glob(os.path.join(path, "Categories", "Categories1", "*")))[:30]
#
#     test_x = sorted(glob(os.path.join(path, "Categories1", "Categories", "*")))[:5]
#     test_y = sorted(glob(os.path.join(path, "Categories1", "Categories1", "*")))[:5]
#
#     kf = KFold(n_splits=n_splits)
#     train_indices, valid_indices = list(kf.split(train_x))[current_fold]
#
#     train_x = [train_x[i] for i in train_indices]
#     train_y = [train_y[i] for i in train_indices]
#
#     valid_x = [train_x[i] for i in valid_indices]
#     valid_y = [train_y[i] for i in valid_indices]
#
#     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


""" Load and split the dataset """
def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "Training", "Training", "*")))[:30]
    train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))[:30]

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)

    valid_x = sorted(glob(os.path.join(path, "Validation", "Training", "*")))[:5]
    valid_y = sorted(glob(os.path.join(path, "Validation", "Categories", "*")))[:5]

    return (train_x, train_y), (valid_x, valid_y), (test_x[:100], test_y[:100])


def grayscale_to_rgb(mask, classes, colormap):
    h, w = mask.shape[:2]
    mask = mask.astype(np.int32)
    output1 = []

    for i, pixel in enumerate(mask.flatten()):
        output1.append(colormap[pixel])

    output1 = np.reshape(output1, (h, w, 3))
    return output1

def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = cv2.resize(pred, (image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST)
    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    cv2.imwrite(save_image , pred)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directory for storing files """
    create_dir("results/rgb")
    create_dir("output_single_mask")

    """ Hyperparameters """
    IMG_H = 512
    IMG_W = 512

    path = "HBX_dataset"
    model_path = os.path.join("files", "model.h5")

    CLASSES, COLORMAP = get_colormap()
    NUM_CLASSES = len(CLASSES)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_u2net(num_classes=NUM_CLASSES)
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # """ Load the dataset """
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(path)
    # print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}")
    # print("")


    test_x=["/home/ml2/Desktop/Vscode/U-Net/U2net/annotation_dataset/validation/im/0103_pm_022.png"]
    test_y=["/home/ml2/Desktop/Vscode/U-Net/U2net/annotation_dataset/validation/im/0103_pm_022.png"]


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
        mask = cv2.resize(mask, (IMG_W, IMG_H) ,interpolation=cv2.INTER_NEAREST)

        onehot_mask = []
        for color in COLORMAP:
            cmap = np.all(np.equal(mask, color), axis=-1)
            onehot_mask.append(cmap)
        onehot_mask = np.stack(onehot_mask, axis=-1)
        onehot_mask = np.argmax(onehot_mask, axis=-1)
        onehot_mask = onehot_mask.astype(np.int32)

        """ Prediction """
        with torch.no_grad():
            pred = model(image)[0]
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = pred[0].astype(np.float32)

        """ Saving the prediction """
        save_image_path = f"results/rgb/{name}.png"
        save_image= f"output_single_mask/{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        print(save_image_path)

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


