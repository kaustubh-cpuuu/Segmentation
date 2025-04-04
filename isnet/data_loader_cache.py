from __future__ import print_function, division
import numpy as np
import random
from copy import deepcopy
import json
from tqdm import tqdm
from skimage import io
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import logging
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms


def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")

    name_im_gt_list = []
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        logging.info(f"--->>> {flag} dataset {i}/{len(datasets)} {datasets[i]['name']} <<<---")

        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])

        # img_name_dict[im_dirs[i][0]] = tmp_im_list
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
        logging.info(f"-im- {datasets[i]['name']} {datasets[i]['im_dir']}: {len(tmp_im_list)}")

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            logging.info(f"-gt- {datasets[i]['name']} {datasets[i]['gt_dir']}: No Ground Truth Found")

            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]

            # lbl_name_dict[im_dirs[i][0]] = tmp_gt_list
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))
            logging.info(f"-gt- {datasets[i]['name']} {datasets[i]['gt_dir']}: {len(tmp_gt_list)}")

        if flag=="train": ## combine multiple training sets into one dataset
            if len(name_im_gt_list)==0:
                name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                        "im_path":tmp_im_list,
                                        "gt_path":tmp_gt_list,
                                        "im_ext":datasets[i]["im_ext"],
                                        "gt_ext":datasets[i]["gt_ext"],
                                        "cache_dir":datasets[i]["cache_dir"]})
            else:
                name_im_gt_list[0]["dataset_name"] = name_im_gt_list[0]["dataset_name"] + "_" + datasets[i]["name"]
                name_im_gt_list[0]["im_path"] = name_im_gt_list[0]["im_path"] + tmp_im_list
                name_im_gt_list[0]["gt_path"] = name_im_gt_list[0]["gt_path"] + tmp_gt_list
                if datasets[i]["im_ext"]!=".jpg" or datasets[i]["gt_ext"]!=".png":
                    print("Error: Please make sure all you images and ground truth masks are in jpg and png format respectively !!!")
                    logging.info("Error: Please make sure all you images and ground truth masks are in jpg and png format respectively !!!")
                    exit()
                name_im_gt_list[0]["im_ext"] = ".jpg"
                name_im_gt_list[0]["gt_ext"] = ".png"
                name_im_gt_list[0]["cache_dir"] = os.sep.join(datasets[i]["cache_dir"].split(os.sep)[0:-1])+os.sep+name_im_gt_list[0]["dataset_name"]
        else: ## keep different validation or inference datasets as separate ones
            name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                    "im_path":tmp_im_list,
                                    "gt_path":tmp_gt_list,
                                    "im_ext":datasets[i]["im_ext"],
                                    "gt_ext":datasets[i]["gt_ext"],
                                    "cache_dir":datasets[i]["cache_dir"]})

    return name_im_gt_list
def create_dataloaders(name_im_gt_list, cache_size=[], cache_boost=True, my_transforms=[],tran=[] , batch_size=1, shuffle=False):

    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets
    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for i in range(0,len(name_im_gt_list)):
        gos_dataset = GOSDatasetCache([name_im_gt_list[i]],
                                      cache_size = cache_size,
                                      cache_path = name_im_gt_list[i]["cache_dir"],
                                      cache_boost = cache_boost,
                                      transform = transforms.Compose(my_transforms))
        gos_dataloaders.append(DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        gos_datasets.append(gos_dataset)




    return gos_dataloaders, gos_datasets

def im_reader(im_path):
    return io.imread(im_path)

# def im_preprocess(im,size):
#     if len(im.shape) < 3:
#         im = im[:, :, np.newaxis]
#     if im.shape[2] == 1:
#         im = np.repeat(im, 3, axis=2)
#     im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
#     im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
#     if(len(size)<2):
#         return im_tensor, im.shape[0:2]
#     else:
#         im_tensor = torch.unsqueeze(im_tensor,0)
#         im_tensor = F.upsample(im_tensor, size, mode="bilinear")
#         im_tensor = torch.squeeze(im_tensor,0)
#
#     return im_tensor.type(torch.uint8), im.shape[0:2]
# def gt_preprocess(gt,size):
#     if len(gt.shape) > 2:
#         gt = gt[:, :, 0]
#
#     gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8),0)
#
#     if(len(size)<2):
#         return gt_tensor.type(torch.uint8), gt.shape[0:2]
#     else:
#         gt_tensor = torch.unsqueeze(torch.tensor(gt_tensor, dtype=torch.float32),0)
#         gt_tensor = F.upsample(gt_tensor, size, mode="bilinear")
#         gt_tensor = torch.squeeze(gt_tensor,0)
#
#     return gt_tensor.type(torch.uint8), gt.shape[0:2]


def im_preprocess(im, size=None):
    # Ensure the image has 3 channels
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)

    # Convert image to PyTorch tensor
    im_tensor = torch.tensor(im, dtype=torch.float32)
    im_tensor = im_tensor.permute(2, 0, 1)  # Adjust dimensions to CxHxW for PyTorch

    # Add padding if the image is not square
    height, width = im_tensor.shape[1], im_tensor.shape[2]
    if height != width:
        # Determine the size of padding needed to make the image square
        diff = abs(height - width) // 2
        padding = (diff, diff, 0, 0)  # Padding for left/right or top/bottom
        if height < width:
            padding = (0, 0, diff, diff)  # Adjust padding if height is less than width
        # Apply padding
        im_tensor = F.pad(im_tensor, padding, "constant", 0)  # Add black padding

    # Resize if a target size is specified
    if size is not None and len(size) == 2:
        im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

    return im_tensor.type(torch.uint8), (height, width)


def gt_preprocess(gt, size=None):
    # Ensure GT is in a single-channel format
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    # Convert GT to PyTorch tensor
    gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

    # Add padding if the GT is not square
    height, width = gt_tensor.shape[1], gt_tensor.shape[2]
    if height != width:
        # Determine the size of padding needed to make the GT square
        diff = abs(height - width) // 2
        padding = (diff, diff, 0, 0)  # Padding for left/right or top/bottom
        if height < width:
            padding = (0, 0, diff, diff)  # Adjust padding if height is less than width
        # Apply padding
        gt_tensor = F.pad(gt_tensor, padding, "constant", 0)  # Add black padding

    # Resize if a target size is specified
    if size is not None and len(size) == 2:
        gt_tensor = F.interpolate(gt_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

    return gt_tensor.type(torch.uint8), (height, width)

class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

            # print(f"arter flip , image shape {image.shape}")

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class GOSRandomPatchExtraction(object):
    def __init__(self, patch_size=1024, logging=False):
        self.patch_size = patch_size
        self.logging = logging  # Add an option to enable/disable logging

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        original_shape = image.shape

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        cropped_shape = image.shape
        # if self.logging:
            # print(f"Original shape: {original_shape}, Cropped to: {cropped_shape}")

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}



# class GOSRandomPatchExtraction(object):
#     def __init__(self, patch_size=1024, logging=False):
#         self.patch_size = patch_size
#         self.logging = logging
#
#     def __call__(self, sample):
#         imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
#         original_shape = image.shape
#
#         # Ensure we get two different patches
#         i1, j1, h1, w1 = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
#         image_patch1 = TF.crop(image, i1, j1, h1, w1)
#         label_patch1 = TF.crop(label, i1, j1, h1, w1)
#
#         # Attempt to get a second distinct patch
#         attempts = 0
#         while attempts < 10:  # Limit attempts to avoid infinite loop
#             i2, j2, h2, w2 = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
#             # Check if the second patch is different from the first
#             if not (i1 == i2 and j1 == j2):
#                 image_patch2 = TF.crop(image, i2, j2, h2, w2)
#                 label_patch2 = TF.crop(label, i2, j2, h2, w2)
#                 break
#             attempts += 1
#
#         if self.logging:
#             print(f"Original shape: {original_shape}")
#             print(f"Patch 1 cropped to: {image_patch1.shape}")
#             print(f"Patch 2 cropped to: {image_patch2.shape}")
#
#         # Return two patches instead of one
#         return {'imidx': imidx,
#                 'image_patch1': image_patch1, 'label_patch1': label_patch1,
#                 'image_patch2': image_patch2, 'label_patch2': label_patch2,
#                 'shape': shape}



class GOSRandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSRandomCropWithRatio(object):
    def __init__(self, size=[288, 288], foreground_ratio=0.6):
        self.size = size
        self.foreground_ratio = foreground_ratio

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        max_attempts = 100
        for _ in range(max_attempts):
            top = torch.randint(0, h - new_h, (1,)).item()
            left = torch.randint(0, w - new_w, (1,)).item()

            label_crop = label[:, top:top + new_h, left:left + new_w]
            foreground_pixels = torch.sum(label_crop == 255).item()  # Use PyTorch's sum method
            total_pixels = new_h * new_w

            if foreground_pixels / total_pixels >= self.foreground_ratio:
                image = image[:, top:top + new_h, left:left + new_w]
                label = label_crop
                return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}

        # Fallback strategy: crop the middle of the image
        mid_h, mid_w = h // 2, w // 2
        top = max(0, mid_h - new_h // 2)
        left = max(0, mid_w - new_w // 2)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSResize(object):
    def __init__(self,size=[512,512]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # import time
        # start = time.time()

        original_image, original_label = image.clone(), label.clone()

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear' , align_corners=True),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear' , align_corners=True),dim=0)


        # print("time for resize: ", time.time()-start)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

        # return {'imidx': imidx,
        #         'original_image': original_image, 'original_label': original_label,
        #         'image': image, 'label': label, 'shape': shape}

"""Added"""
class GOSColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0 , prob=0.5):
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.prob = prob
    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        image = self.color_jitter(image)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}


class GOSDatasetCache(Dataset):

    def __init__(self, name_im_gt_list, cache_size=[], cache_path='./cache', cache_file_name='dataset.json', cache_boost=False, transform=None ):


        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        self.cache_boost_name = ""

        self.cache_boost = cache_boost
        # self.ims_npy = None
        # self.gts_npy = None

        ## cache all the images and ground truth into a single pytorch tensor
        self.ims_pt = None
        self.gts_pt = None

        ## we will cache the npy as well regardless of the cache_boost
        # if(self.cache_boost):
        self.cache_boost_name = cache_file_name.split('.json')[0]

        self.transform = transform


        self.dataset = {}

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])



        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_shp"] = []
        self.dataset["gt_shp"] = []
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.dataset["ims_pt_dir"] = ""
        self.dataset["gts_pt_dir"] = ""

        self.dataset = self.manage_cache(dataset_names)


    def manage_cache(self,dataset_names):
        if not os.path.exists(self.cache_path): # create the folder for cache
            os.makedirs(self.cache_path)
        cache_folder = os.path.join(self.cache_path, "_".join(dataset_names)+"_"+"x".join([str(x) for x in self.cache_size]))
        if not os.path.exists(cache_folder): # check if the cache files are there, if not then cache
            return self.cache(cache_folder)
        return self.load_cache(cache_folder)

    def cache(self,cache_folder):
        os.mkdir(cache_folder)
        cached_dataset = deepcopy(self.dataset)

        # ims_list = []
        # gts_list = []
        ims_pt_list = []
        gts_pt_list = []
        for i, im_path in tqdm(enumerate(self.dataset["im_path"]), total=len(self.dataset["im_path"])):

            im_id = cached_dataset["im_name"][i]
            print("im_path: ", im_path)
            logging.info(f"im_path: {im_path} ")
            im = im_reader(im_path)
            im, im_shp = im_preprocess(im,self.cache_size)
            im_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_im.pt")
            torch.save(im,im_cache_file)

            cached_dataset["im_path"][i] = im_cache_file
            if(self.cache_boost):
                ims_pt_list.append(torch.unsqueeze(im,0))
            # ims_list.append(im.cpu().data.numpy().astype(np.uint8))

            gt = np.zeros(im.shape[0:2])
            if len(self.dataset["gt_path"])!=0:
                gt = im_reader(self.dataset["gt_path"][i])
            gt, gt_shp = gt_preprocess(gt,self.cache_size)
            gt_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_gt.pt")
            torch.save(gt,gt_cache_file)
            if len(self.dataset["gt_path"])>0:
                cached_dataset["gt_path"][i] = gt_cache_file
            else:
                cached_dataset["gt_path"].append(gt_cache_file)
            if(self.cache_boost):
                gts_pt_list.append(torch.unsqueeze(gt,0))
            # gts_list.append(gt.cpu().data.numpy().astype(np.uint8))

            # im_shp_cache_file = os.path.join(cache_folder,im_id + "_im_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["im_shp"].append(im_shp)
            # self.dataset["im_shp"].append(im_shp)

            # shp_cache_file = os.path.join(cache_folder,im_id + "_gt_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["gt_shp"].append(gt_shp)
            # self.dataset["gt_shp"].append(gt_shp)

        if(self.cache_boost):
            cached_dataset["ims_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_ims.pt')
            cached_dataset["gts_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_gts.pt')
            self.ims_pt = torch.cat(ims_pt_list,dim=0)
            self.gts_pt = torch.cat(gts_pt_list,dim=0)
            torch.save(torch.cat(ims_pt_list,dim=0),cached_dataset["ims_pt_dir"])
            torch.save(torch.cat(gts_pt_list,dim=0),cached_dataset["gts_pt_dir"])

        try:
            json_file = open(os.path.join(cache_folder, self.cache_file_name),"w")
            json.dump(cached_dataset, json_file)
            json_file.close()
        except Exception:
            raise FileNotFoundError("Cannot create JSON")
        return cached_dataset

    def load_cache(self, cache_folder):
        json_file = open(os.path.join(cache_folder,self.cache_file_name),"r")
        dataset = json.load(json_file)
        json_file.close()
        ## if cache_boost is true, we will load the image npy and ground truth npy into the RAM
        ## otherwise the pytorch tensor will be loaded
        if(self.cache_boost):
            # self.ims_npy = np.load(dataset["ims_npy_dir"])
            # self.gts_npy = np.load(dataset["gts_npy_dir"])
            self.ims_pt = torch.load(dataset["ims_pt_dir"], map_location='cpu')
            self.gts_pt = torch.load(dataset["gts_pt_dir"], map_location='cpu')
        return dataset

    def __len__(self):
        return len(self.dataset["im_path"])



    def __getitem__(self, idx):

        im = None
        gt = None
        if(self.cache_boost and self.ims_pt is not None):

            # start = time.time()

            im = self.ims_pt[idx]#.type(torch.float32)
            gt = self.gts_pt[idx]#.type(torch.float32)
            # print(idx, 'time for pt loading: ', time.time()-start)

        else:
            # import time
            # start = time.time()
            # print("tensor***")
            im_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["im_path"][idx].split(os.sep)[-2:]))
            im = torch.load(im_pt_path)#(self.dataset["im_path"][idx])
            gt_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["gt_path"][idx].split(os.sep)[-2:]))
            gt = torch.load(gt_pt_path)#(self.dataset["gt_path"][idx])
            # print(idx,'time for tensor loading: ', time.time()-start)


        im_shp = self.dataset["im_shp"][idx]
        # print("time for loading im and gt: ", time.time()-start)

        # start_time = time.time()
        im = torch.divide(im,255.0)
        gt = torch.divide(gt,255.0)
        # print(idx, 'time for normalize torch divide: ', time.time()-start_time)
        #
        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "shape": torch.from_numpy(np.array(im_shp)),
        }
        #
        # sample = {
        #     "imidx": torch.from_numpy(np.array(idx)),
        #     "original_image": im.clone(),  # Clone to keep the original
        #     "original_label": gt.clone(),  # Clone to keep the original
        #     "image": im,
        #     "label": gt,
        #     "shape": torch.from_numpy(np.array(im_shp)),
        # }

        if self.transform:
            sample = self.transform(sample)
        return sample






















