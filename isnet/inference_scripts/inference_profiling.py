import os
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet_DenseNet_121 import ISNetDIS
import cv2
from glob import glob
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import cProfile
import pstats
import io as StringIO

def apply_crf(original_image, mask_probabilities):
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)
    unary = unary_from_softmax(np.stack((1 - mask_probabilities, mask_probabilities), axis=0))
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=original_image, compat=10)
    print(d)
    q = d.inference(1)
    map = np.argmax(q, axis=0).reshape(original_image.shape[0], original_image.shape[1])
    return map

def post_process_segmentation(segmentation):
    segmentation = segmentation.cpu().numpy().squeeze()
    _, binary_mask = cv2.threshold(segmentation, 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels_im = cv2.connectedComponents((closed_mask * 255).astype(np.uint8))
    min_size = 100
    cleaned_mask = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            cleaned_mask[labels_im == label] = 1
    return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)

def process_images():
    dataset_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset"
    model_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\testing_checkpoint\\test_resolu\\chache_size_1024\\IS-Netisnet.pth"
    result_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset_result"
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im_list = glob(dataset_path + "\\*.jpg") + glob(dataset_path + "\\*.jpeg") + glob(dataset_path + "\\*.png") + glob(
        dataset_path + "\\*.bmp") + glob(dataset_path + "\\*.tiff")

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("Processing image: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=True).type(torch.float32)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if torch.cuda.is_available():
                image = image.cuda()
            result = net(image)

            result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)
            result_numpy = result.cpu().numpy().squeeze()
            original_image = im.astype(np.uint8)

            crf_result = apply_crf(original_image, result_numpy)
            post_processed_result = post_process_segmentation(torch.from_numpy(crf_result).float())
            result_3ch = post_processed_result.repeat(3, 1, 1)
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)

            im_name = os.path.split(im_path)[-1].split('.')[0]
            io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)

def main():
    pr = cProfile.Profile()
    pr.enable()

    process_images()

    pr.disable()
    s = StringIO.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    main()
