import os
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet_DenseNet_121 import ISNetDIS
import cv2  # Import OpenCV
from glob import glob
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
import torch

def apply_crf(original_image, mask_probabilities):

    original_image = np.ascontiguousarray(original_image)
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)
    unary = unary_from_softmax(np.stack((1 - mask_probabilities, mask_probabilities), axis=0))
    d.setUnaryEnergy(unary)

    # This is a simplified example. You might need to experiment with these parameters
    # d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # d.addPairwiseBilateral(sxy=50, srgb=150, rgbim=original_image, compat=10)    #srg=150

    d.addPairwiseGaussian(sxy=(2,2), compat=3 , kernel=dcrf.DIAG_KERNEL , normalization=dcrf.NORMALIZE_BEFORE )
    d.addPairwiseBilateral(sxy=4, srgb=20 , rgbim=original_image, compat=10 , kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_BEFORE)
    #                        )


    # Run inference
    q = d.inference(5)
    map = np.argmax(q, axis=0).reshape(original_image.shape[0], original_image.shape[1])
    return map


# """With Morphological operation"""
def post_process_segmentation(segmentation):
    segmentation = segmentation.cpu().numpy().squeeze()
    _, binary_mask = cv2.threshold(segmentation, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological closing to fill small holes
    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents((closed_mask * 255).astype(np.uint8))
    min_size = 100
    cleaned_mask = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            cleaned_mask[labels_im == label] = 1

    return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)

# def find_bounding_box(mask):
#     """Find the bounding box (min_row, min_col, max_row, max_col) for white pixels."""
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#     min_row, max_row = np.where(rows)[0][[0, -1]]
#     min_col, max_col = np.where(cols)[0][[0, -1]]
#
#     return min_row, min_col, max_row, max_col

def find_bounding_box(mask, expansion=10):
    """
    Find the bounding box (min_row, min_col, max_row, max_col) for white pixels and expand it.
    Parameters:
        mask (np.array): The mask array where the bounding box is to be found.
        expansion (int): The number of pixels to expand the bounding box on each side.
    Returns:
        tuple: The expanded bounding box (min_row, min_col, max_row, max_col).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Expand the bounding box by 'expansion' pixels, ensuring it doesn't go out of the image bounds
    min_row = max(min_row - expansion, 0)
    min_col = max(min_col - expansion, 0)
    max_row = min(max_row + expansion, mask.shape[0] - 1)
    max_col = min(max_col + expansion, mask.shape[1] - 1)

    return min_row, min_col, max_row, max_col

def crop_to_bbox(image, bbox):
    """Crop an image to the specified bounding box."""
    min_row, min_col, max_row, max_col = bbox
    return image[min_row:max_row + 1, min_col:max_col + 1]


def stitch_back(original, crop, bbox):
    """Stitch the cropped part back into the original image."""
    min_row, min_col, max_row, max_col = bbox
    stitched = original.copy()
    stitched[min_row:max_row + 1, min_col:max_col + 1] = crop
    return stitched

if __name__ == "__main__":
    dataset_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset"
    model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\2500_hbx.pth"
    result_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset_result"
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    # im_list = glob(os.path.join(dataset_path, "*.jpg"))
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

            """For features and non-features"""
            result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)
            # result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='nearest'), 0)


            """" Prediction from the model """
            # post_processed_result = post_process_segmentation(result)
            # result_3ch = post_processed_result.repeat(3, 1, 1)
            # result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            # result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8
            # im_name = os.path.split(im_path)[-1].split('.')[0]
            # io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)


            result_numpy = result.cpu().numpy().squeeze()
            original_image = im.astype(np.uint8)
            bbox = find_bounding_box(result_numpy > 0.5)

            # Crop the image and mask according to the bounding box.
            cropped_image = crop_to_bbox(original_image, bbox)
            cropped_mask = crop_to_bbox(result_numpy, bbox)


            """Cropped Part save"""
            # im_name = os.path.split(im_path)[-1].split('.')[0]
            # cropped_image_path = os.path.join(result_path, f"{im_name}_cropped_image.png")
            # cropped_mask_path = os.path.join(result_path, f"{im_name}_cropped_mask.png")
            # io.imsave(cropped_image_path, cropped_image)
            # io.imsave(cropped_mask_path, (cropped_mask * 255).astype(np.uint8))  # Convert mask to uint8 for saving


            # Apply CRF on the cropped parts.
            crf_result = apply_crf(cropped_image, cropped_mask)
            post_processed_cropped_result = post_process_segmentation(torch.from_numpy(crf_result).float())
            stitched_result = stitch_back(result_numpy, post_processed_cropped_result.numpy(), bbox)
            stitched_result_3ch = np.repeat(stitched_result[:, :, np.newaxis], 3, axis=2)
            stitched_result_3ch_uint8 = (stitched_result_3ch * 255).astype(np.uint8)

            im_name = os.path.split(im_path)[-1].split('.')[0]
            io.imsave(os.path.join(result_path, im_name + "_processed.png"), stitched_result_3ch_uint8)




