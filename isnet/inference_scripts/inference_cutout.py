import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS

if __name__ == "__main__":
    dataset_path = "C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\your_dataset"
    model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\isnet-general-use.pth"
    result_path = "C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\your_dataset_result"
    cutout_path = "C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\cutout_results"  # Path for cutout images
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.jpeg") + \
              glob(dataset_path + "/*.png") + glob(dataset_path + "/*.bmp") + \
              glob(dataset_path + "/*.tiff")

    if not os.path.exists(cutout_path):
        os.makedirs(cutout_path)

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            if torch.cuda.is_available():
                image = image.cuda()
            result = net(image)
            result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)

            # Convert mask to binary (0 or 1)
            mask = result > 0.5  # Threshold to create a binary mask
            mask_3ch = mask.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()

            # Apply mask to the image
            im_cutout = im * mask_3ch

            im_name = os.path.split(im_path)[-1].split('.')[0]
            io.imsave(os.path.join(cutout_path, im_name + "_cutout.png"), im_cutout.astype(np.uint8))
