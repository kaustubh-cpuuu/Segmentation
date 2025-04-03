import os
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
from isnet import ISNetDIS


if __name__ == "__main__":
    dataset_path="C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\your_dataset"
    model_path   =  r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Netisnet.pth"
    result_path="C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\your_dataset_result"
    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear" , align_corners=True ).type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0 )
            ma = torch.max(result)
            mi = torch.min(result)

            result_3ch = result.repeat(3, 1, 1)
            # result = (result-mi)/(ma-mi)
            # im_name=im_path.split('/')[-1].split('.')[0]
            # io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))

            im_name = os.path.split(im_path)[-1].split('.')[0]
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8

            io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)









""" FOR HBX DATASET"""


# import os
# import time
# import numpy as np
# from skimage import io
# import time
# from glob import glob
# from tqdm import tqdm
# import torch, gc
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision.transforms.functional import normalize
# from isnet import ISNetDIS
#
# if __name__ == "__main__":
#     dataset_path="C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\hbx_dataset"  # Your dataset path
#     model_path=    r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\isnet-general-use.pth"  # the model path
#     result_path="C:\\Users\ml2au\PycharmProjects\Background_removal\DIS\demo_datasets\hbx_result"  #The folder path that you want to save the results
#     input_size=[1024,1024]
#     net=ISNetDIS()
#
#     if torch.cuda.is_available():
#         net.load_state_dict(torch.load(model_path))
#         net=net.cuda()
#     else:
#         net.load_state_dict(torch.load(model_path,map_location="cpu"))
#     net.eval()
#     im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")
#     with torch.no_grad():
#         for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
#             print("im_path: ", im_path)
#             im = io.imread(im_path)
#
#             # Ensure the image has at least one color channel
#             if len(im.shape) < 3:
#                 im = im[:, :, np.newaxis]
#
#             # If the image is grayscale (1 channel), convert it to a 3-channel image
#             if im.shape[2] == 1:
#                 im = np.repeat(im, 3, axis=2)
#
#             # Resize and preprocess the image before feeding it to the model
#             im_shp = im.shape[0:2]
#             im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
#             im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear",
#                                       align_corners=True).type(torch.uint8)
#             image = torch.divide(im_tensor, 255.0)
#             image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
#             if torch.cuda.is_available():
#                 image=image.cuda()
#             result=net(image)
#             result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0 )
#             ma = torch.max(result)
#             mi = torch.min(result)
#
#             result_3ch = result.repeat(3, 1, 1)
#             # result = (result-mi)/(ma-mi)
#             # im_name=im_path.split('/')[-1].split('.')[0]
#             # io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
#
#             im_name = os.path.split(im_path)[-1].split('.')[0]
#             result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
#             result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8
#
#             io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)








