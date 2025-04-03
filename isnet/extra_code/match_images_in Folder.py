#
# import os
# import hashlib
# from glob import glob
#
#
# def compute_md5(filename):
#     """ Compute MD5 hash of a file """
#     hash_md5 = hashlib.md5()
#     with open(filename, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()
#
#
# def match_and_remove_images_by_name(folder1, folder2):
#     """ Match images based on filename (without extension) between two folders and remove unmatched images """
#
#     # Get list of image files from both directories
#     folder1_images = glob(os.path.join(folder1, '*'))
#     folder2_images = glob(os.path.join(folder2, '*'))
#
#     # Extract filenames without extension
#     folder1_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder1_images}
#     folder2_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder2_images}
#
#     # Identify unmatched images
#     unmatched_folder1 = set(folder1_names.keys()) - set(folder2_names.keys())
#     unmatched_folder2 = set(folder2_names.keys()) - set(folder1_names.keys())
#
#     # Remove unmatched images
#     for name in unmatched_folder1:
#         os.remove(folder1_names[name])
#
#     for name in unmatched_folder2:
#         os.remove(folder2_names[name])
#
#     return len(unmatched_folder1), len(unmatched_folder2)
#
# # The function now matches images based on their filenames, excluding the file extension.
#
# folder1_removed, folder2_removed = match_and_remove_images_by_name("ww", "CelebA-HQ-img/0.jpg")
# print(f"Removed {folder1_removed} unmatched images from folder1 and {folder2_removed} from folder2.")
#
#
import os
import shutil
from glob import glob

def match_and_remove_images_by_name(folder1, folder2, unmatched_folder1_path, unmatched_folder2_path):
    """
    Match images based on filename (without extension) between two folders and
    move unmatched images to specified folders.
    """

    # Ensure the unmatched folders exist
    os.makedirs(unmatched_folder1_path, exist_ok=True)
    os.makedirs(unmatched_folder2_path, exist_ok=True)

    # Get list of image files from both directories
    folder1_images = glob(os.path.join(folder1, '*'))
    folder2_images = glob(os.path.join(folder2, '*'))

    # Extract filenames without extension
    folder1_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder1_images}
    folder2_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder2_images}

    # Identify unmatched images
    unmatched_folder1 = set(folder1_names.keys()) - set(folder2_names.keys())
    unmatched_folder2 = set(folder2_names.keys()) - set(folder1_names.keys())

    # Move unmatched images
    for name in unmatched_folder1:
        shutil.move(folder1_names[name], unmatched_folder1_path)

    for name in unmatched_folder2:
        shutil.move(folder2_names[name], unmatched_folder2_path)

    return len(unmatched_folder1), len(unmatched_folder2)

# The function now moves unmatched images to the specified unmatched folders.

unmatched_folder1_path = 'new_jwellery_dataset/gt'  # Update with the actual path
unmatched_folder2_path = 'new_jwellery_dataset/gt2'  # Update with the actual path

folder1_removed, folder2_removed = match_and_remove_images_by_name(
    "Product/images",
    "Product/mask",
    unmatched_folder1_path,
    unmatched_folder2_path
)

print(f"Moved {folder1_removed} unmatched images from folder1 to {unmatched_folder1_path} and {folder2_removed} from folder2 to {unmatched_folder2_path}.")
