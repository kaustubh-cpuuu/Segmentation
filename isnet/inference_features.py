"""For Features Generation"""

import os
os.environ["KMP_DUPLICATE_LIB_  OK"] = "TRUE"
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet_2048 import ISNetDIS
import cv2
import cv2.ximgproc as xip
import time
import skimage.restoration as restoration


def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Apply unsharp masking to sharpen the image."""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    blurred = cv2.GaussianBlur(gray_image, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray_image, 1 + strength, blurred, -strength, 0)
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened


if __name__ == "__main__":
    dataset_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/little_icon"
    model_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/gpu_itr_18000_traLoss_0.0832_traTarLoss_0.0106_valLoss_0.0841_valTarLoss_0.0131_maxF1_0.2336_mae_0.0048_time_0.024593.pth"
    result_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/output"
    input_size = [2048, 2048]
    net = ISNetDIS()


    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Load model weights on CPU
    # device = torch.device('cpu')  # Force CPU
    # net.load_state_dict(torch.load(model_path, map_location=device))
    # net.eval()

    net.eval()

    im_list = (
        glob(dataset_path + "/*.jpg")
        + glob(dataset_path + "/*.JPG")
        + glob(dataset_path + "/*.jpeg")
        + glob(dataset_path + "/*.JPEG")
        + glob(dataset_path + "/*.png")
        + glob(dataset_path + "/*.PNG")
        + glob(dataset_path + "/*.bmp")
        + glob(dataset_path + "/*.BMP")
        + glob(dataset_path + "/*.tiff")
        + glob(dataset_path + "/*.TIFF")
    )
    start_time = time.time()
    with torch.no_grad():

        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):

            im = io.imread(im_path)

            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(
                torch.unsqueeze(im_tensor, 0),
                input_size,
                mode="bilinear",
                align_corners=False,
            ).type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            if torch.cuda.is_available():
                image = image.cuda()
            # with torch.cuda.amp.autocast():
            #     result = net(image)
            result = net(image)

            result = torch.squeeze(F.upsample(result[0][0], im_shp, mode="bilinear"), 0)

            ma = torch.max(result)
            mi = torch.min(result)

            result_3ch = result.repeat(3, 1, 1)

            im_name = os.path.split(im_path)[-1].split(".")[0]
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(
                np.uint8
            )  # Convert to uint8
            print(result_3ch_np_uint8.dtype)

            """unsharpen mask"""
            # result_3ch_np_uint8 = unsharp_mask(result_3ch_np_uint8, sigma=1.0, strength=1.5)

            """Bilateral Filtter"""
            # result_3ch_np_uint8 = cv2.bilateralFilter(result_3ch_np_uint8, d=9 , sigmaColor=75, sigmaSpace=75 )

            # """Gaussian_Blur"""
            # start_time = time.time()
            # blurred = cv2.GaussianBlur(result_3ch_np_uint8, (5, 5), 0)
            # result_3ch_np_uint8 = cv2.addWeighted(result_3ch_np_uint8, 1.5, blurred, -0.5, 0)
            # end_time = time.time()
            # print(f"processing time per image for Gaussianblur {end_time - start_time:.6f} Seconds")

            """GuidedFilter"""
            # start_time = time.time()
            # result_3ch_np_uint8 = xip.guidedFilter(guide=result_3ch_np_uint8, src=result_3ch_np_uint8, radius=20, eps=0.01)
            # result_3ch_np_uint8 = xip.guidedFilter(guide=result_3ch_np_uint8, src=result_3ch_np_uint8, radius=20, eps=0.01)
            # result_3ch_np_uint8 = xip.guidedFilter(guide=result_3ch_np_uint8, src=result_3ch_np_uint8, radius=20, eps=0.01)
            # result_3ch_np_uint8 = xip.guidedFilter(guide=result_3ch_np_uint8, src=result_3ch_np_uint8, radius=20, eps=0.01)
            # end_time = time.time()
            # processing_time = end_time - start_time
            # print(f"Processing time per image for guided filter: {processing_time:.6f} seconds")

            # result_3ch_np_uint8 = cv2.bilateralFilter(result_3ch_np_uint8, d=9 , sigmaColor=75, sigmaSpace=75 )

            "Post processing in 3 steps for Minor"
            # mask = result_3ch_np_uint8 > 0
            # result_3ch_np_uint8[mask] = np.clip(
            #     result_3ch_np_uint8[mask].astype(np.int16) + 50, 0, 255
            # ).astype(np.uint8)

            # mask = (result_3ch_np_uint8 > 0) & (result_3ch_np_uint8 >= 50)
            # result_3ch_np_uint8[mask] = np.clip(
            #     result_3ch_np_uint8[mask].astype(np.int16) + 20, 0, 255
            # ).astype(np.uint8)

            # _, result_3ch_np_uint8 = cv2.threshold(result_3ch_np_uint8, 64, 255, cv2.THRESH_BINARY)
            # kernel = np.ones((3, 3), np.uint8)
            # binary_mask = cv2.dilate(result_3ch_np_uint8, kernel, iterations=1)


            # Replace all occurrences of [254, 254, 254] with [255, 255, 255]
            # mask = np.all(result_3ch_np_uint8 == [254, 254, 254], axis=-1)
            # result_3ch_np_uint8[mask] = [255, 255, 255]

            io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)
            end_time = time.time()
        total = end_time - start_time
        print(f"total time {total} ")
