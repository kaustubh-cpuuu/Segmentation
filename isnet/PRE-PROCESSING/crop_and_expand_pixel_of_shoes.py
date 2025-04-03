# import os
# import cv2
# import numpy as np

# def crop_and_expand_images_based_on_mask(image_folder, mask_folder, output_image_folder, output_mask_folder,
#           te output directories if they don't exist
#     os.mak                               expand_pixels=20):
#     # Creaedirs(output_image_folder, exist_ok=True)
#     os.makedirs(output_mask_folder, exist_ok=True)

#     # List all files in the image folder
#     image_files = os.listdir(image_folder)

#     for image_file in image_files:
#         # Load image
#         image_path = os.path.join(image_folder, image_file)
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Error loading image: {image_path}")
#             continue

#         # Load corresponding mask
#         mask_file = get_corresponding_mask_file(image_file, mask_folder)
#         if not mask_file:
#             continue

#         mask_path = os.path.join(mask_folder, mask_file)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Error loading mask: {mask_path}")
#             continue

#         # Find contours in the mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Calculate the bounding box that encompasses all contours
#         if contours:
#             all_x, all_y, all_w, all_h = cv2.boundingRect(np.vstack(contours))
#             x_expanded = max(0, all_x - expand_pixels)
#             y_expanded = max(0, all_y - expand_pixels)
#             w_expanded = min(image.shape[1] - x_expanded, all_w + 2 * expand_pixels)
#             h_expanded = min(image.shape[0] - y_expanded, all_h + 2 * expand_pixels)

#             # Crop the image and mask using the calculated bounding box
#             cropped_image = image[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]
#             cropped_mask = mask[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

#             # Save cropped image and mask
#             cropped_image_path = os.path.join(output_image_folder, f"{image_file.split('.')[0]}.jpg")
#             cropped_mask_path = os.path.join(output_mask_folder, f"{mask_file.split('.')[0]}.png")
#             cv2.imwrite(cropped_image_path, cropped_image)
#             cv2.imwrite(cropped_mask_path, cropped_mask)

# def get_corresponding_mask_file(image_file, mask_folder):
#     base_name = os.path.splitext(image_file)[0]
#     for mask_format in ['.png', '.jpg', '.jpeg']:  # Add or remove formats as needed
#         mask_file = f"{base_name}{mask_format}"
#         if os.path.exists(os.path.join(mask_folder, mask_file)):
#             return mask_file
#     return None

# if __name__ == "__main__":
#     # Paths to image and mask folders

#     image_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Multiclass/bottom/im"
#     mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Multiclass/bottom/gt"

#     output_image_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Multiclass/im_50"
#     output_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Multiclass/gt_50"

#     # Crop and expand images based on mask
#     crop_and_expand_images_based_on_mask(image_folder, mask_folder, output_image_folder, output_mask_folder,
#                                          expand_pixels=50)




import os
import cv2
import numpy as np
import multiprocessing

def process_image(image_file, image_folder, mask_folder, output_image_folder, output_mask_folder, expand_pixels):
    # Load image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Load corresponding mask
    mask_file = get_corresponding_mask_file(image_file, mask_folder)
    if not mask_file:
        return

    mask_path = os.path.join(mask_folder, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask: {mask_path}")
        return

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the bounding box that encompasses all contours
    if contours:
        all_x, all_y, all_w, all_h = cv2.boundingRect(np.vstack(contours))
        x_expanded = max(0, all_x - expand_pixels)
        y_expanded = max(0, all_y - expand_pixels)
        w_expanded = min(image.shape[1] - x_expanded, all_w + 2 * expand_pixels)
        h_expanded = min(image.shape[0] - y_expanded, all_h + 2 * expand_pixels)

        # Crop the image and mask using the calculated bounding box
        cropped_image = image[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]
        cropped_mask = mask[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

        # Save cropped image and mask
        cropped_image_path = os.path.join(output_image_folder, f"{image_file.split('.')[0]}.jpg")
        cropped_mask_path = os.path.join(output_mask_folder, f"{mask_file.split('.')[0]}.png")
        cv2.imwrite(cropped_image_path, cropped_image)
        cv2.imwrite(cropped_mask_path, cropped_mask)

def get_corresponding_mask_file(image_file, mask_folder):
    base_name = os.path.splitext(image_file)[0]
    for mask_format in ['.png', '.jpg', '.jpeg']:  # Add or remove formats as needed
        mask_file = f"{base_name}{mask_format}"
        if os.path.exists(os.path.join(mask_folder, mask_file)):
            return mask_file
    return None

def crop_and_expand_images_multiprocessing(image_folder, mask_folder, output_image_folder, output_mask_folder, expand_pixels=20):
    # Create output directories if they don't exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    # Get number of CPU cores
    num_workers = multiprocessing.cpu_count()
    
    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=num_workers)
    
    tasks = [(image_file, image_folder, mask_folder, output_image_folder, output_mask_folder, expand_pixels) for image_file in image_files]
    pool.starmap(process_image, tasks)
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Paths to image and mask folders
    image_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/im"
    mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/gt"

    output_image_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/imm"
    output_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/gtt"

    # Crop and expand images using multiprocessing
    crop_and_expand_images_multiprocessing(image_folder, mask_folder, output_image_folder, output_mask_folder,
                                           expand_pixels=20)