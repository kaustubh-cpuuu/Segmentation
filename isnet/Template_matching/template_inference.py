# import os
# import time
# import numpy as np
# from skimage import io
# from glob import glob
# from tqdm import tqdm
# import torch
# import torch.nn.functional as F
# from torchvision.transforms.functional import normalize
# from isnet import ISNetDIS
# import cv2

# def run_inference(dataset_path, model_path, result_path, input_size):
#     net = ISNetDIS()
    
#     if torch.cuda.is_available():
#         net.load_state_dict(torch.load(model_path))
#         net = net.cuda()
#     else:
#         net.load_state_dict(torch.load(model_path, map_location="cpu"))
    
#     net.eval()
#     im_list = glob(dataset_path + "/*.[jpJgE][pP][gG]") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.bmp") + glob(dataset_path + "/*.tiff")
    
#     with torch.no_grad():
#         for im_path in tqdm(im_list, total=len(im_list)):
#             im = io.imread(im_path)

#             if len(im.shape) < 3:
#                 im = im[:, :, np.newaxis]
#             im_shp = im.shape[0:2]
#             im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
#             im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
#             image = torch.divide(im_tensor, 255.0)
#             image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
#             if torch.cuda.is_available():
#                 image = image.cuda()

#             result = net(image)
#             result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
#             result_3ch = result.repeat(3, 1, 1)
                                                
#             im_name = os.path.split(im_path)[-1].split('.')[0]
#             result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
#             result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)

#             io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)
#     return result_path

# def run_template_matching(original_images_folder, cropped_images_folder, output_directory):
#     os.makedirs(output_directory, exist_ok=True)

#     original_images = [f for f in os.listdir(original_images_folder) if f.endswith(('.jpg', '.png'))]
#     cropped_images = [f for f in os.listdir(cropped_images_folder) if f.endswith(('.jpg', '.png'))]

#     if len(original_images) != len(cropped_images):
#         raise ValueError("The number of original images and cropped images must match.")

#     for original_image_name, cropped_image_name in zip(original_images, cropped_images):
#         original_image_path = os.path.join(original_images_folder, original_image_name)
#         cropped_image_path = os.path.join(cropped_images_folder, cropped_image_name)

#         original_image = cv2.imread(original_image_path)
#         cropped_image = cv2.imread(cropped_image_path)

#         if original_image is None or cropped_image is None:
#             continue

#         gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#         gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

#         orb = cv2.ORB_create()
#         keypoints1, descriptors1 = orb.detectAndCompute(gray_original, None)
#         keypoints2, descriptors2 = orb.detectAndCompute(gray_cropped, None)

#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         matches = bf.match(descriptors1, descriptors2)

#         if not matches:
#             continue

#         matches = sorted(matches, key=lambda x: x.distance)

#         points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#         points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

#         H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

#         if H is None:
#             continue

#         aligned_cropped = cv2.warpPerspective(cropped_image, H, (original_image.shape[1], original_image.shape[0]))
#         transparent_canvas = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
#         aligned_cropped_with_alpha = cv2.cvtColor(aligned_cropped, cv2.COLOR_BGR2BGRA)

#         mask = cv2.cvtColor(aligned_cropped, cv2.COLOR_BGR2GRAY)
#         _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             x, y, w, h = cv2.boundingRect(contours[0])
#             transparent_canvas[y:y+h, x:x+w] = aligned_cropped_with_alpha[y:y+h, x:x+w]
#             output_path = os.path.join(output_directory, f'aligned_{original_image_name}')
#             cv2.imwrite(output_path, transparent_canvas)

# def main():
#     # Set paths
#     dataset_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/gt"
#     model_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/all_in_one_2048.pth"
#     result_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/inference_result"
#     input_size = [1024, 1024]

#     # Run inference
#     inference_results = run_inference(dataset_path, model_path, result_path, input_size)

#     # Paths for template matching
#     original_images_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/im"
#     cropped_images_folder = dataset_path
#     output_directory = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/res"

#     # Run template matching
#     run_template_matching(original_images_folder, cropped_images_folder, output_directory)

# if __name__ == "__main__":
#     main()




import os
import time
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS
import cv2

def run_inference(dataset_path, model_path, result_path, input_size):
    net = ISNetDIS()
    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    net.eval()
    im_list = glob(dataset_path + "/*.[jpJgE][pP][gG]") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.bmp") + glob(dataset_path + "/*.tiff")
    
    with torch.no_grad():
        for im_path in tqdm(im_list, total=len(im_list)):
            im = io.imread(im_path)

            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if torch.cuda.is_available():
                image = image.cuda()

            result = net(image)
            result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
            result_3ch = result.repeat(3, 1, 1)
                                                
            im_name = os.path.split(im_path)[-1].split('.')[0]
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)

            # Save the binary mask (grayscale)
            io.imsave(os.path.join(result_path, im_name + "_mask.png"), result_3ch_np_uint8[:,:,0])  # Save single-channel mask
    return result_path

def run_template_matching(original_images_folder, cropped_images_folder, mask_images_folder, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    original_images = [f for f in os.listdir(original_images_folder) if f.endswith(('.jpg', '.png'))]
    cropped_images = [f for f in os.listdir(cropped_images_folder) if f.endswith(('.jpg', '.png'))]
    mask_images = [f for f in os.listdir(mask_images_folder) if f.endswith('_mask.png')]

    if len(original_images) != len(cropped_images) or len(cropped_images) != len(mask_images):
        raise ValueError("The number of original images, cropped images, and mask images must match.")

    for original_image_name, cropped_image_name, mask_image_name in zip(original_images, cropped_images, mask_images):
        original_image_path = os.path.join(original_images_folder, original_image_name)
        cropped_image_path = os.path.join(cropped_images_folder, cropped_image_name)
        mask_image_path = os.path.join(mask_images_folder, mask_image_name)

        original_image = cv2.imread(original_image_path)
        cropped_image = cv2.imread(cropped_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        if original_image is None or cropped_image is None or mask_image is None:
            continue

        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray_original, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray_cropped, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        if not matches:
            continue

        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        if H is None:
            continue

        # Warp the mask image to align with the original image
        aligned_mask = cv2.warpPerspective(mask_image, H, (original_image.shape[1], original_image.shape[0]))

        # Create a transparent canvas and add the aligned mask
        transparent_canvas = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        
        # Create an RGBA mask (white on the mask area, transparent elsewhere)
        mask_rgba = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2BGRA)
        # mask_rgba[aligned_mask > 0] = (255, 255, 255, 255)  # White where mask exists

        # Find the bounding box of the aligned mask
        contours, _ = cv2.findContours(aligned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            transparent_canvas[y:y+h, x:x+w] = mask_rgba[y:y+h, x:x+w]  # Paste the mask in the corresponding location

            # Save the result
            output_path = os.path.join(output_directory, f'{original_image_name}')
            cv2.imwrite(output_path, transparent_canvas)

def main():
    # Set paths
    dataset_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/gt"
    model_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/all_in_one_2048.pth"
    result_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/inference_result"
    input_size = [1024, 1024]

    # Run inference
    inference_results = run_inference(dataset_path, model_path, result_path, input_size)

    # Paths for template matching
    original_images_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/im"
    cropped_images_folder = dataset_path
    mask_images_folder = result_path
    output_directory = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/res"

    # Run template matching with binary masks
    run_template_matching(original_images_folder, cropped_images_folder, mask_images_folder, output_directory)

if __name__ == "__main__":
    main()
