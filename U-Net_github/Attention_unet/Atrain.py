import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cv2
import time
import torch
import scipy.io
import random
import datetime
import numpy as np
from PIL import Image
from glob import glob
import torch.nn as nn
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from Autils import seeding, create_dir, print_and_save, shuffling, epoch_time
from Amodel import attention_unet
from sklearn.model_selection import train_test_split, KFold

import torch.nn as nn
import torch

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP


""" Load and split the dataset """
def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "training", "im", "*")))[:80]
    train_y = sorted(glob(os.path.join(path, "training", "gt", "*")))[:80]

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)

    valid_x = sorted(glob(os.path.join(path, "validation", "im", "*")))[:20]
    valid_y = sorted(glob(os.path.join(path, "validation", "gt", "*")))[:20]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




def get_colormap():
    colormap = [
        [0, 0, 0],       # Background
        [0, 0, 255],     # T-shirt
        [0, 228, 255] ,   # Skin
        [255, 216, 0],   # shoes
        [0, 255, 0],     # Bottom

    ]

    classes = [
        "Background",
        "T-shirt",
        "skin",
        "shoes",
        "Bottom",

    ]

    return classes, colormap



class DATASET(Dataset):
    def __init__(self,path, images_path, masks_path, size, transform=None, normalize=False):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.normalize = normalize
        self.n_samples = len(images_path)

        self.CLASSES, self.COLORMAP = get_colormap()

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)

        if self.normalize == True:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_COLOR)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = image.astype(np.float32)

        """ Mask """
        mask = cv2.resize(mask, self.size)

        if self.normalize == True:
            output_mask = mask
        else:
            output_mask = []
            for color in self.COLORMAP:
                cmap = np.all(np.equal(mask, color), axis=-1)
                output_mask.append(cmap)
            output_mask = np.stack(output_mask, axis=-1)
            output_mask = output_mask.astype(np.float64)
            output_mask = np.argmax(output_mask, axis=-1)

        return image, output_mask

    def __len__(self):
        return self.n_samples

""" Without mix-precision training """
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad()
        y0 = model(x)
        loss = loss_fn(y0, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

""" With mix-precision training """
# def train(model, loader, optimizer, loss_fn, device):
#     model.train()
#     epoch_loss = 0.0
#     scaler = torch.cuda.amp.GradScaler()
#
#     for x, y in loader:
#         x = x.to(device, dtype=torch.float32)
#         y = y.to(device, dtype=torch.long)
#
#         optimizer.zero_grad()
#         with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=False):
#             y0 = model(x)
#             loss = loss_fn(y0, y)
#
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         epoch_loss += loss.item()
#
#     epoch_loss = epoch_loss/len(loader)
#     return epoch_loss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            y0 = model(x)
            loss = loss_fn(y0, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training22 logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")


    """ Hyperparameters """
    IMG_H = 512
    IMG_W = 512

    size = (IMG_W, IMG_H)
    batch_size = 2
    num_epochs = 50
    lr = 1e-3
    early_stopping_patience = 5
    checkpoint_path = "files/checkpoint.pth"

    """Data Path"""
    path = "/home/ml2/Desktop/Vscode/U-Net/Attention_unet/annotation_dataset"

    CLASSES, COLORMAP = get_colormap( )
    NUM_CLASSES = len(CLASSES)

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Num Classes: {NUM_CLASSES}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(path)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])


    """ Dataset and loader """
    train_dataset = DATASET(path,train_x, train_y, size, transform=transform, normalize=False)
    valid_dataset = DATASET(path,valid_x, valid_y, size, transform=None, normalize=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = attention_unet(num_classes=NUM_CLASSES)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    loss_name = "CrossEntropy Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training22 the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.5f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.5f}\n'
        print_and_save(train_log_path, data_str)



