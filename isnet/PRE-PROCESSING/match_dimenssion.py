# import os
# import shutil
# from PIL import Image



# def check_dimensions(image_path, mask_path):
#     # Load images
#     image = Image.open(image_path)
#     mask = Image.open(mask_path)

#     # Get   dimensions
#     image_size = image.size
#     mask_size = mask.size

#     return image_size == mask_size
 

# def move_files(file_path, destination_dir):
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
#     shutil.move(file_path, os.path.join(destination_dir, os.path.basename(file_path)))



# def compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir):
#     # Get list of image and mask files
#     image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('jpg', 'png', 'jpeg'))}
#     mask_files = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.endswith(('jpg', 'png', 'jpeg'))}

#     # Common file names
#     common_files = image_files & mask_files

#     for file_name in common_files:
#         image_path = os.path.join(images_dir, file_name + '.jpg')  # or '.png', '.jpeg'
#         mask_path = os.path.join(masks_dir, file_name + '.png')  # or '.png', '.jpeg'

#         if not os.path.exists(image_path) or not os.path.exists(mask_path):
#             print(f"Missing file: {file_name}")
#             continue

#         if check_dimensions(image_path, mask_path):
#             print(f"{file_name}: Dimensions match")
#         else:
#             print(f"{file_name}: Dimensions do not match")
#             # Move mismatched files to separate folders
#             move_files(image_path, mismatched_images_dir)
#             move_files(mask_path, mismatched_masks_dir)



# # Replace with your directories
# images_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Dataset_2048_pretrained/training/im"
# masks_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Dataset_2048_pretrained/training/gt"

# mismatched_images_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Dataset_2048_pretrained/training/imm"
# mismatched_masks_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Dataset_2048_pretrained/training/gtt"

# compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir)




import os
import shutil
from PIL import Image, UnidentifiedImageError


def check_dimensions(image_path, mask_path):
    """Check if image and mask have the same dimensions."""
    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        return image.size == mask.size
    except UnidentifiedImageError:
        return None  # Return None to indicate an unreadable file


def move_files(file_path, destination_dir):
    """Move files to the specified directory."""
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    shutil.move(file_path, os.path.join(destination_dir, os.path.basename(file_path)))


def compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir, corrupted_dir):
    """Compare image and mask dimensions and move mismatched or corrupted files."""
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.endswith(('jpg', 'png', 'jpeg'))}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if f.endswith(('jpg', 'png', 'jpeg'))}

    common_files = image_files.keys() & mask_files.keys()

    for file_name in common_files:
        image_path = os.path.join(images_dir, image_files[file_name])
        mask_path = os.path.join(masks_dir, mask_files[file_name])

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Missing file: {file_name}")
            continue

        result = check_dimensions(image_path, mask_path)

        if result is None:  # If file is unreadable
            print(f"Corrupted or unreadable file: {file_name}")
            move_files(image_path, corrupted_dir)
            move_files(mask_path, corrupted_dir)
        elif not result:
            print(f"{file_name}: Dimensions do not match")
            move_files(image_path, mismatched_images_dir)
            move_files(mask_path, mismatched_masks_dir)
        else:
            print(f"{file_name}: Dimensions match")


# Replace with your directories
images_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/im"
masks_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt"

mismatched_images_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/im1"
mismatched_masks_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt1"

corrupted_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/im_corrupted"

compare_image_mask_dimensions(images_dir, masks_dir, mismatched_images_dir, mismatched_masks_dir, corrupted_dir)
