import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
from Rutils import seeding, create_dir, print_and_save, shuffling, epoch_time
from Rmodel import build_u2net
from sklearn.model_selection import KFold, train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

"""K-Fold Cross Categories"""
# def load_dataset(path, split=0.1, n_splits=10, current_fold=0):
#     train_x = sorted(glob(os.path.join(path, "Training", "Training", "*")))[:20]
#     train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))[:20]
#
#     valid_x = sorted(glob(os.path.join(path, "Validation", "Training", "*")))[:5]
#     valid_y = sorted(glob(os.path.join(path, "Validation", "Categories", "*")))[:5]
#
#     test_x = sorted(glob(os.path.join(path, "Validation", "Training", "*")))[:5]
#     test_y = sorted(glob(os.path.join(path, "Validation", "Categories", "*")))[:5]
#
#     kf = KFold(n_splits=n_splits)
#     train_indices, valid_indices = list(kf.split(train_x))[current_fold]
#
#     train_x = [train_x[i] for i in train_indices]
#     train_y = [train_y[i] for i in train_indices]
#
#     # test_x = [train_x[i] for i in valid_indices]
#     # test_y = [train_y[i] for i in valid_indices]
#
#     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

""" Load and split the dataset """
def load_dataset(path, split=0.001):
    train_x = sorted(glob(os.path.join(path, "training", "im", "*")))[:20]
    train_y = sorted(glob(os.path.join(path, "training", "gt", "*")))[:20]

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)

    valid_x = sorted(glob(os.path.join(path, "validation", "im", "*")))[:10]
    valid_y = sorted(glob(os.path.join(path, "validation", "gt", "*")))[:10]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


# def get_colormap():
#     colormap = [
#         [0, 0, 0],       # Background
#         [0, 0, 255],     # T-shirt
#         [0, 228, 255] ,   # Skin
#         [255, 216, 0],   # shoes
#         [221, 119, 0],  # Jacket
#         [0, 128, 0 ],    # Bag
#         [221, 170, 51],  # Gloves
#         [85, 0, 0],      # Top
#         [0, 255, 0],     # Bottom
#         [0, 198, 163],   # Accessory
#         [0, 85, 255],    # Shirt
#         [0, 61, 10]     # Caps

#     ]

#     classes = [
#         "Background",
#         "T-shirt",
#         "skin",
#         "shoes",
#         "Jacket",
#         "Bag",
#         "Gloves",
#         "Top",
#         "Bottom",
#         "Accessory",
#         "Shirt",
#         "Caps"

#     ]

#     return classes, colormap


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
    def __init__(self, path,images_path, masks_path, size, transform=None, normalize=False):
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
        mask = cv2.resize(mask, self.size )  #interpolation=cv2.INTER_LINEAR




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
# def train(model, loader, optimizer, loss_fn, device , clip_value=2.0):
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad()
        y0, y1, y2, y3, y4, y5, y6 = model(x)
        loss = loss_fn(y0, y) + loss_fn(y1, y) + loss_fn(y2, y) + loss_fn(y3, y) + loss_fn(y4, y) + loss_fn(y5, y) + loss_fn(y6, y)
        loss.backward()

        """Gradient clipping """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

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
#             y0, y1, y2, y3, y4, y5, y6 = model(x)
#             loss = loss_fn(y0, y) + loss_fn(y1, y) + loss_fn(y2, y) + loss_fn(y3, y) + loss_fn(y4, y) + loss_fn(y5, y) + loss_fn(y6, y)
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

            y0, y1, y2, y3, y4, y5, y6 = model(x)
            loss = loss_fn(y0, y) + loss_fn(y1, y) + loss_fn(y2, y) + loss_fn(y3, y) + loss_fn(y4, y) + loss_fn(y5, y) + loss_fn(y6, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    train_losses = []
    valid_losses = []

    """ Training logfile """
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
    num_epochs = 10
    lr = 1e-3
    early_stopping_patience = 15
    checkpoint_path = "files/checkpoint.pth"
    checkpoint1_path = "files/checkpoint1.pth"
    path = "/home/ml2/Desktop/Vscode/U-Net/U2net/annotation_dataset"

    CLASSES, COLORMAP = get_colormap()
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
    prob= 0.3
    transform = A.Compose([
        A.Perspective(scale=(0.05, 0.1), p=prob),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=prob),
        A.GaussNoise(var_limit=(10.0, 50.0), p=prob),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=prob),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=prob),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=prob),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-10, 10), p=prob),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=prob),
        A.Rotate(limit=35, p=prob),
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.CoarseDropout(p=prob, max_holes=10, max_height=32, max_width=32),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=prob),

    ], is_check_shapes=False)

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

    device = torch.device('cuda')
    model = build_u2net(num_classes=NUM_CLASSES)
    """Mulit Gpu Usage"""
    # model = nn.DataParallel(model)  # Wrap the model with DataParallel
    # model = model.to(device)  # Move the wrapped model to the GP
    model = model.to(device)

    """TO train It on Pretrain model file """
    # pretrained_weight_path="files/checkpoint.pth"
    # if os.path.exists(pretrained_weight_path):
    #     model.load_state_dict(torch.load(pretrained_weight_path))
    #     print(f"Loaded Pretrained weight from {pretrained_weight_path}")
    # else:
    #     print(f"Pretrained weight not found at {pretrained_weight_path}. Training from Scratch")


    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)
    weight_decay_value = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay_value)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, verbose=True)

    # loss_fn = nn.CrossEntropyLoss()
    # loss_name = "CrossEntropy Loss"

    loss_fn = smp.losses.DiceLoss(mode='multiclass')
    loss_name = "Dice Loss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        # train_loss = train(model, train_loader, optimizer, loss_fn, device , clip_value=0.5) # Gradient clipping
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        """Saving the model on every epoch """
        checkpoint_filename = "files/checkpoint_epoch_latest.pth"  # Constant filename to overwrite
        torch.save(model.state_dict(), checkpoint_filename)
        data_str = f"Epoch {epoch + 1}: Model checkpoint saved/overwritten as {checkpoint_filename}"
        print_and_save(train_log_path, data_str)

        # """saving the model on train loss """
        if train_loss<best_train_loss:
            data_str = f"Train Loss improved from  {best_train_loss:2.4f} to {train_loss:2.4f}. saving checkpoint: {checkpoint1_path}"
            print_and_save(train_log_path, data_str)
            best_train_loss=train_loss
            torch.save(model.state_dict() , checkpoint1_path)

        """ Saving the model on val loss"""
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

