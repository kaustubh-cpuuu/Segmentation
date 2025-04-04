import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu , load_checkpoint
from networks import U2NET

device = "cuda"

image_dir = "input_images"
result_dir = "output_images"
# checkpoint_path = os.path.join("prev_checkpoints", "cloth_segm.pth")
checkpoint_path = os.path.join("results/test/checkpoints", "itr_00001000_u2net.pth")

do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=6)
checkpoint = torch.load(checkpoint_path)
# net = load_checkpoint_mgpu(net, checkpoint_path)
net = load_checkpoint(net, checkpoint_path)

net = net.to(device)
net = net.eval()

palette = get_palette(7)

images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))
for image_name in images_list:
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    print(np.unique(output_arr))

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
        output_img.putpalette(palette)
    output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

    pbar.update(1)

pbar.close()


