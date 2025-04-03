import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from DIS.isnet import ISNetDIS

if __name__ == "__main__":
    dataset_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\imm"
    model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Netisnet.pth"
    result_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\yolo_output"
    input_size = [1024, 1024]
    resize_dim = 2048
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.JPG") + glob(dataset_path + "/*.jpeg") + glob(dataset_path + "/*.JPEG") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.PNG") + glob(dataset_path + "/*.bmp") + glob(dataset_path + "/*.BMP") + glob(dataset_path + "/*.tiff") + glob(dataset_path + "/*.TIFF")

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):

            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]

            original_height, original_width = im.shape[0:2]

            if original_height >= original_width:
                aspect_ratio = original_width / original_height
                new_height = resize_dim
                new_width = int(new_height * aspect_ratio)
            else:
                aspect_ratio = original_height / original_width
                new_width = resize_dim
                new_height = int(new_width * aspect_ratio)

            # Resize image to the new dimensions
            im_resized = F.interpolate(torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0), [new_height, new_width], mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy()

            im_shp = im_resized.shape[0:2]
            im_tensor = torch.tensor(im_resized, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            if torch.cuda.is_available():
                image = image.cuda()

            result = net(image)
            result = torch.squeeze(F.interpolate(result[0][0], [new_height, new_width], mode='bilinear', align_corners=False), 0)
            result_3ch = result.repeat(3, 1, 1)

            im_name = os.path.split(im_path)[-1].split('.')[0]
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8

            io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)






