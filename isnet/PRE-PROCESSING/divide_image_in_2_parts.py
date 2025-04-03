# from PIL import Image
# import os
# def crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks):
#     # Load image and mask
#     image = Image.open(image_path)
#     mask = Image.open(mask_path)
#
#     # Determine the longest side and calculate midpoint
#     width, height = image.size
#     if width > height:
#         midpoint = width // 2
#         box1 = (0, 0, midpoint, height)
#         box2 = (midpoint, 0, width, height)
#     else:
#         midpoint =     height // 2
#         box1 = (0, 0, width, midpoint)
#         box2 = (0, midpoint, width, height)
#
#     # Crop the image and mask
#     image1, image2 = image.crop(box1), image.crop(box2)
#     mask1, mask2 = mask.crop(box1), mask.crop(box2)
#
#     # Save the cropped images and masks
#     base_filename = os.path.basename(image_path)
#     filename_without_ext = os.path.splitext(base_filename)[0]
#
#     image1.save(os.path.join(save_dir_images, f"{filename_without_ext}_1.jpg"))
#     image2.save(os.path.join(save_dir_images, f"{filename_without_ext}_2.jpg"))
#     mask1.save(os.path.join(save_dir_masks, f"{filename_without_ext}_1.png"))
#     mask2.save(os.path.join(save_dir_masks, f"{filename_without_ext}_2.png"))
#
#
# def process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks):
#     # Make sure the save directories exist
#     os.makedirs(save_dir_images, exist_ok=True)
#     os.makedirs(save_dir_masks, exist_ok=True)
#
#     # Process each image in the images directory
#     for image_filename in os.listdir(images_dir):
#         image_path = os.path.join(images_dir, image_filename)
#
#         # Construct the corresponding mask path with PNG extension
#         filename_without_ext = os.path.splitext(image_filename)[0]
#         mask_path = os.path.join(masks_dir, f"{filename_without_ext}.png")
#
#         # Check if the mask file exists
#         if not os.path.exists(mask_path):
#             print(f"Mask for {image_filename} not found, skipping.")
#             continue
#
#         print(f"Processing {image_filename}...")
#         crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks)
#
# # Define your directories
# images_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\onepiece_hbx\traininig\im"
# masks_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\onepiece_hbx\traininig\gt"
#
# save_dir_images = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\onepiece_hbx\traininig\imm"
# save_dir_masks = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\onepiece_hbx\traininig\gtt"
#
# process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks)


import os
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_save_image(image, mask, save_dir_images, save_dir_masks, base_filename, divide):
    """
    Helper function to save the image and mask, either as a whole or in two parts based on the 'divide' flag.
    """
    filename_without_ext = os.path.splitext(base_filename)[0]

    if divide:
        # Determine the longest side and calculate midpoint
        width, height = image.size
        if width > height:
            midpoint = width // 2
            box1 = (0, 0, midpoint, height)
            box2 = (midpoint, 0, width, height)
        else:
            midpoint = height // 2
            box1 = (0, 0, width, midpoint)
            box2 = (0, midpoint, width, height)

        # Crop the image and mask
        image1, image2 = image.crop(box1), image.crop(box2)
        mask1, mask2 = mask.crop(box1), mask.crop(box2)

        # Save the cropped images and masks
        image1.save(os.path.join(save_dir_images, f"{filename_without_ext}_1.jpg"))
        image2.save(os.path.join(save_dir_images, f"{filename_without_ext}_2.jpg"))
        mask1.save(os.path.join(save_dir_masks, f"{filename_without_ext}_1.png"))
        mask2.save(os.path.join(save_dir_masks, f"{filename_without_ext}_2.png"))

    else:
        # Save the whole image and mask
        image.save(os.path.join(save_dir_images, f"{filename_without_ext}.jpg"))
        mask.save(os.path.join(save_dir_masks, f"{filename_without_ext}.png"))

    logging.info(f"Processed {base_filename}, divide: {divide}")


def crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks):
    try:
        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Determine if the image should be divided
        width, height = image.size
        divide = width >= 1024 or height >= 1024

        # Save image and mask, possibly divided
        process_and_save_image(image, mask, save_dir_images, save_dir_masks, os.path.basename(image_path), divide)

    except Exception as e:
        logging.error(f"Error processing {image_path} and {mask_path}: {e}")


def process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks):
    # Make sure the save directories exist
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_masks, exist_ok=True)

    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # Process each image in the images directory
    for image_filename in os.listdir(images_dir):
        if os.path.splitext(image_filename)[1].lower() not in image_extensions:
            logging.warning(f"Skipping unsupported file type: {image_filename}")
            continue

        image_path = os.path.join(images_dir, image_filename)

        # Construct the corresponding mask path with PNG extension
        filename_without_ext = os.path.splitext(image_filename)[0]
        mask_path = os.path.join(masks_dir, f"{filename_without_ext}.png")

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            logging.warning(f"Mask for {image_filename} not found, skipping.")
            continue

        logging.info(f"Processing {image_filename}...")
        crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks)


# Define your directories
images_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\imm"
masks_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\gtt"
save_dir_images = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\immm"
save_dir_masks = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\gttt"

process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks)
