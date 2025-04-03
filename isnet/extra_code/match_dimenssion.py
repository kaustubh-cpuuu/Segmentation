import os
import shutil
from PIL import Image


def check_dimensions(image_path, mask_path):
    # Load images
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Get dimensions
    image_size = image.size
    mask_size = mask.size

    return image_size == mask_size


def move_files(file_path, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    shutil.move(file_path, os.path.join(destination_dir, os.path.basename(file_path)))


def compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir):
    # Get list of image and mask files
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('jpg', 'png', 'jpeg'))}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.endswith(('jpg', 'png', 'jpeg'))}

    # Common file names
    common_files = image_files & mask_files

    for file_name in common_files:
        image_path = os.path.join(images_dir, file_name + '.jpg')  # or '.png', '.jpeg'
        mask_path = os.path.join(masks_dir, file_name + '.png')  # or '.png', '.jpeg'

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Missing file: {file_name}")
            continue

        if check_dimensions(image_path, mask_path):
            print(f"{file_name}: Dimensions match")
        else:
            print(f"{file_name}: Dimensions do not match")
            # Move mismatched files to separate folders
            move_files(image_path, mismatched_images_dir)
            move_files(mask_path, mismatched_masks_dir)


# Replace with your directories
images_dir = r'match_images_in Folder.pyC:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes_dataset\training\im'
masks_dir = r'C:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes_dataset\training\gt'
mismatched_images_dir = r'C:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes_dataset\training\imm'
mismatched_masks_dir = r'C:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes_dataset\training\gtt'

compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir)
