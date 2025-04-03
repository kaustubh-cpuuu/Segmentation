
# import os
# import random
# import shutil
# from concurrent.futures import ThreadPoolExecutor

# def move_file_pair(img_src, mask_src, img_dst, mask_dst):
#     """Function to move a single image-mask pair."""
#     try:
#         shutil.copy(img_src, img_dst)
#         shutil.copy(mask_src, mask_dst)
#     except Exception as e:
#         print(f"Error moving files: {e}")

# def move_random_images_and_masks(input_img_folder, input_mask_folder, output_img_folder, output_mask_folder, num_samples):
#     os.makedirs(output_img_folder, exist_ok=True)
#     os.makedirs(output_mask_folder, exist_ok=True)

#     # Get lists of image and mask filenames
#     img_files = list(os.scandir(input_img_folder))
#     mask_files = list(os.scandir(input_mask_folder))

#     # Create base name sets for matching
#     img_base_names = {os.path.splitext(f.name)[0]: f.name for f in img_files}
#     mask_base_names = {os.path.splitext(f.name)[0]: f.name for f in mask_files}

#     # Find common base names
#     common_base_names = list(set(img_base_names.keys()) & set(mask_base_names.keys()))

#     # Limit to num_samples
#     num_samples = min(num_samples, len(common_base_names))
#     selected_base_names = random.sample(common_base_names, num_samples)



#     tasks = []
#     for base_name in selected_base_names:
#         img_src = os.path.join(input_img_folder, img_base_names[base_name])
#         mask_src = os.path.join(input_mask_folder, mask_base_names[base_name])
#         img_dst = os.path.join(output_img_folder, img_base_names[base_name])
#         mask_dst = os.path.join(output_mask_folder, mask_base_names[base_name])
#         tasks.append((img_src, mask_src, img_dst, mask_dst))

#     # Use ThreadPool for faster I/O
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         executor.map(lambda t: move_file_pair(*t), tasks)

#     print(f"Moved {num_samples} image-mask pairs.")

# # Example usage
# input_img_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/original/im"
# input_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/original/gt"

# output_img_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/hair/im"
# output_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/hair/gt"

# num_samples = 116

# move_random_images_and_masks(input_img_folder, input_mask_folder, output_img_folder, output_mask_folder, num_samples)





import os
import random
import shutil
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor



def is_valid_image(image_path):
    """Check if the file is a valid image."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image integrity
        return True
    except UnidentifiedImageError:
        return False
    except Exception as e:
        print(f"Error verifying image {image_path}: {e}")
        return False


def move_file_pair(img_src, mask_src, img_dst, mask_dst, corrupted_dir):
    """Move image-mask pairs or handle corrupted files."""
    try:
        # Check if either file is corrupted
        if not is_valid_image(img_src) or not is_valid_image(mask_src):
            print(f"Corrupted file detected: {os.path.basename(img_src)} or {os.path.basename(mask_src)}")
            os.makedirs(corrupted_dir, exist_ok=True)
            shutil.move(img_src, os.path.join(corrupted_dir, os.path.basename(img_src)))
            shutil.move(mask_src, os.path.join(corrupted_dir, os.path.basename(mask_src)))
        else:
            shutil.move(img_src, img_dst)
            shutil.move(mask_src, mask_dst)
    except Exception as e:
        print(f"Error moving files: {e}")


def move_random_images_and_masks(input_img_folder, input_mask_folder, output_img_folder, output_mask_folder, corrupted_dir, num_samples):
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # Get lists of image and mask filenames
    img_files = list(os.scandir(input_img_folder))
    mask_files = list(os.scandir(input_mask_folder))

    # Create base name sets for matching
    img_base_names = {os.path.splitext(f.name)[0]: f.name for f in img_files}
    mask_base_names = {os.path.splitext(f.name)[0]: f.name for f in mask_files}

    # Find common base names
    common_base_names = list(set(img_base_names.keys()) & set(mask_base_names.keys()))

    # Limit to num_samples
    num_samples = min(num_samples, len(common_base_names))
    selected_base_names = random.sample(common_base_names, num_samples)

    tasks = []
    for base_name in selected_base_names:
        img_src = os.path.join(input_img_folder, img_base_names[base_name])
        mask_src = os.path.join(input_mask_folder, mask_base_names[base_name])
        img_dst = os.path.join(output_img_folder, img_base_names[base_name])
        mask_dst = os.path.join(output_mask_folder, mask_base_names[base_name])
        tasks.append((img_src, mask_src, img_dst, mask_dst, corrupted_dir))

    # Use ThreadPool for faster I/O
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda t: move_file_pair(*t), tasks)

    print(f"Moved {num_samples} image-mask pairs. Corrupted files were moved to {corrupted_dir}")


# Example usage

input_img_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/im"
input_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt"

output_img_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/validation/im"
output_mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/validation/gt"

corrupted_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/validation/im_corrupted"

num_samples = 400


move_random_images_and_masks(input_img_folder, input_mask_folder, output_img_folder, output_mask_folder, corrupted_dir, num_samples)
