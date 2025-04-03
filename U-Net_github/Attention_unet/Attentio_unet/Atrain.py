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
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time
from Amodel import attention_unet


""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Load and split the dataset """
def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "Training", "Training", "*")))[:100]
    train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))[:100]

    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)

    valid_x = sorted(glob(os.path.join(path, "Validation", "Training", "*")))[:40]
    valid_y = sorted(glob(os.path.join(path, "Validation", "Categories", "*")))[:40]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def get_colormap():
    colormap = [
        [0, 0, 0],  # B
        [255, 0, 0],  # C
        [255, 228, 0],  # S
        [0, 255, 0],  # H
        [0, 216, 255],  # M
        [0, 0, 255],
    ]

    classes = [
        "Background",
        "Clothes",
        "Skin",
        "Pant",
        "Shoes",
        "hand"
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
    num_epochs = 30
    lr = 1e-4
    early_stopping_patience = 5
    checkpoint_path = "files/checkpoint.pth"

    """Data Path"""
    path = "dataaaa"

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

























# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# import random
# import time
# import datetime
# import numpy as np
# from PIL import Image
# import albumentations as A
# import cv2
# from glob import glob
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import scipy.io
# from sklearn.model_selection import train_test_split
# from Autils import seeding, create_dir, print_and_save, shuffling, epoch_time
# from Amodel import attention_unet
# from sklearn.model_selection import KFold
#
# """ Global parameters """
# global IMG_H
# global IMG_W
# global NUM_CLASSES
# global CLASSES
# global COLORMAP
#
# """K-Fold Cross Validation1"""
# def load_dataset(path, split=0.1, n_splits=10, current_fold=0):
#     train_x = sorted(glob(os.path.join(path, "Training1", "Training1", "*")))[:50]
#     train_y = sorted(glob(os.path.join(path, "Training1", "Categories1", "*")))[:50]
#
#     test_x = sorted(glob(os.path.join(path, "Validation1", "Training1", "*")))[:5]
#     test_y = sorted(glob(os.path.join(path, "Validation1", "Categories1", "*")))[:5]
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
#
#
# """ Load and split the dataset """
# # def load_dataset(path, split=0.2):
# #     train_x = sorted(glob(os.path.join(path, "Training1", "Training1", "*")))[:6000]
# #     train_y = sorted(glob(os.path.join(path, "Training1", "Categories1", "*")))[:6000]
# #
# #     train_x, test_x = train_test_split(train_x, test_size=split, random_state=42)
# #     train_y, test_y = train_test_split(train_y, test_size=split, random_state=42)
# #
# #     valid_x = sorted(glob(os.path.join(path, "Validation1", "Training1", "*")))[:1000]
# #     valid_y = sorted(glob(os.path.join(path, "Validation1", "Categories1", "*")))[:1000]
# #
# #     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
#
# def get_colormap(path):
#     mat_path = os.path.join(path, "human_colormap.mat")
#     colormap = scipy.io.loadmat(mat_path)["colormap"]
#     colormap = colormap * 256
#     colormap = colormap.astype(np.uint8)
#     colormap = [[c[2], c[1], c[0]] for c in colormap]
#
#     classes = [
#         "Background",
#         "Hat",
#         "Hair",
#         "Glove",
#         "Sunglasses",
#         "UpperClothes",
#         "Dress",
#         "Coat",
#         "Socks",
#         "Pants",
#         "Torso-skin",
#         "Scarf",
#         "Skirt",
#         "Face",
#         "Left-arm",
#         "Right-arm",
#         "Left-leg",
#         "Right-leg",
#         "Left-shoe",
#         "Right-shoe"
#     ]
#
#     return classes, colormap
#
# class DATASET(Dataset):
#     def __init__(self,path, images_path, masks_path, size, transform=None, normalize=False):
#         super().__init__()
#
#         self.images_path = images_path
#         self.masks_path = masks_path
#         self.size = size
#         self.transform = transform
#         self.normalize = normalize
#         self.n_samples = len(images_path)
#
#         self.CLASSES, self.COLORMAP = get_colormap(path)
#     def __getitem__(self, index):
#         """ Image """
#         image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
#
#         if self.normalize == True:
#             mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
#         else:
#             mask = cv2.imread(self.masks_path[index], cv2.IMREAD_COLOR)
#
#         if self.transform is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]
#
#         """ Image """
#         image = cv2.resize(image, self.size)
#         image = np.transpose(image, (2, 0, 1))
#         image = image / 255.0
#         image = image.astype(np.float32)
#
#         """ Mask """
#         mask = cv2.resize(mask, self.size)
#
#         if self.normalize == True:
#             output_mask = mask
#         else:
#             output_mask = []
#             for color in self.COLORMAP:
#                 cmap = np.all(np.equal(mask, color), axis=-1)
#                 output_mask.append(cmap)
#             output_mask = np.stack(output_mask, axis=-1)
#             output_mask = output_mask.astype(np.float64)
#             output_mask = np.argmax(output_mask, axis=-1)
#
#         return image, output_mask
#
#     def __len__(self):
#         return self.n_samples
#
# """ Without mix-precision training """
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
#         y0 = model(x)
#         loss = loss_fn(y0, y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#
#     epoch_loss = epoch_loss/len(loader)
#     return epoch_loss
#
# """ With mix-precision training """
# # def train(model, loader, optimizer, loss_fn, device):
# #     model.train()
# #     epoch_loss = 0.0
# #     scaler = torch.cuda.amp.GradScaler()
# #
# #     for x, y in loader:
# #         x = x.to(device, dtype=torch.float32)
# #         y = y.to(device, dtype=torch.long)
# #
# #         optimizer.zero_grad()
# #         with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=False):
# #             y0 = model(x)
# #             loss = loss_fn(y0, y)
# #
# #         scaler.scale(loss).backward()
# #         scaler.step(optimizer)
# #         scaler.update()
# #         epoch_loss += loss.item()
# #
# #     epoch_loss = epoch_loss/len(loader)
# #     return epoch_loss
#
# def evaluate(model, loader, loss_fn, device):
#     model.eval()
#     epoch_loss = 0.0
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device, dtype=torch.float32)
#             y = y.to(device, dtype=torch.long)
#
#             y0 = model(x)
#             loss = loss_fn(y0, y)
#             epoch_loss += loss.item()
#
#         epoch_loss = epoch_loss/len(loader)
#     return epoch_loss
#
# if __name__ == "__main__":
#     """ Seeding """
#     seeding(42)
#
#     """ Directories """
#     create_dir("files")
#
#     """ Training1 logfile """
#     train_log_path = "files/train_log.txt"
#     if os.path.exists(train_log_path):
#         print("Log file exists")
#     else:
#         train_log = open("files/train_log.txt", "w")
#         train_log.write("\n")
#         train_log.close()
#
#     """ Record Date & Time """
#     datetime_object = str(datetime.datetime.now())
#     print_and_save(train_log_path, datetime_object)
#     print("")
#
#     def get_lr(optimizer):
#         for param_group in optimizer.param_groups:
#             return param_group['lr']
#
#     """ Hyperparameters """
#     IMG_H = 512
#     IMG_W = 512
#
#     size = (IMG_W, IMG_H)
#     batch_size = 2
#     num_epochs = 10
#     lr = 1e-4
#     early_stopping_patience = 5
#     checkpoint_path = "files/checkpoint.pth"
#     path = "instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing"
#
#     CLASSES, COLORMAP = get_colormap(path)
#     NUM_CLASSES = len(CLASSES)
#
#     data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
#     data_str += f"Num Classes: {NUM_CLASSES}\n"
#     data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
#     print_and_save(train_log_path, data_str)
#
#     """ Dataset """
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(path)
#     train_x, train_y = shuffling(train_x, train_y)
#     data_str = f"Dataset Size:\nTrain: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}\n"
#     print_and_save(train_log_path, data_str)
#
#     """ Data augmentation: Transforms """
#     prob = 0.3
#     transform =  A.Compose([
#         A.Rotate(limit=35, p=prob),
#         A.HorizontalFlip(p=prob),
#         A.VerticalFlip(p=prob),
#         A.CoarseDropout(p=prob, max_holes=10, max_height=32, max_width=32),
#         A.Blur(blur_limit=3, p=prob),
#         A.ElasticTransform(p=prob, alpha=1, sigma=50),
#         A.ChannelShuffle(p=prob)
#     ])
#
#     """ Dataset and loader """
#     train_dataset = DATASET(path,train_x, train_y, size, transform=transform, normalize=False)
#     valid_dataset = DATASET(path,valid_x, valid_y, size, transform=None, normalize=False)
#
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2
#     )
#
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2
#     )
#
#     """ Model """
#     device = torch.device('cuda')
#     model = attention_unet(num_classes=NUM_CLASSES)
#     # model = nn.DataParallel(model)  # Wrap the model with DataParallel
#     # model = model.to(device)  # Move the wrapped model to the GP
#     model = model.to(device)
#
#     """TO train It on Pretrain model file """
#     # pretrained_weight_path="files/checkpoint.pth"
#     # if os.path.exists(pretrained_weight_path):
#     #     model.load_state_dict(torch.load(pretrained_weight_path))
#     #     print(f"Loaded Pretrained weight from {pretrained_weight_path}")
#     # else:
#     #     print(f"Pretrained weight not found at {pretrained_weight_path}. Training1 from Scratch")
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
#     loss_fn = nn.CrossEntropyLoss()
#     loss_name = "CrossEntropy Loss"
#     data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
#     print_and_save(train_log_path, data_str)
#
#     """ Training1 the model """
#     best_valid_loss = float("inf")
#
#     for epoch in range(num_epochs):
#         start_time = time.time()
#
#         train_loss = train(model, train_loader, optimizer, loss_fn, device)
#         valid_loss = evaluate(model, valid_loader, loss_fn, device)
#         scheduler.step(valid_loss)
#
#
#         epoch_losses = []
#         epoch_losses.append((train_loss, valid_loss))
#         # Inside the training loop
#         epoch_losses.append((train_loss, valid_loss))
#
#         current_lr = get_lr(optimizer)
#
#         """ Saving the model """
#         if valid_loss < best_valid_loss:
#             data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
#             print_and_save(train_log_path, data_str)
#             best_valid_loss = valid_loss
#             torch.save(model.state_dict(), checkpoint_path)
#
#         end_time = time.time()
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#         data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
#         data_str += f'\tTrain Loss: {train_loss:.5f}\n'
#         data_str += f'\t Val. Loss: {valid_loss:.5f}\n'
#         data_str += f" Current Learning Rate: {current_lr}\n"
#         print_and_save(train_log_path, data_str)
#
#
#
#
