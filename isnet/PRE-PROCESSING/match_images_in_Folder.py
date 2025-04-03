import os
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count

def move_unmatched_images(args):
    """Helper function to move unmatched images."""
    name, image_path, unmatched_folder_path = args
    shutil.move(image_path, unmatched_folder_path)

def match_and_remove_images_by_name(folder1, folder2, unmatched_folder1_path, unmatched_folder2_path):

    # Ensure the unmatched folders exist
    os.makedirs(unmatched_folder1_path, exist_ok=True)
    os.makedirs(unmatched_folder2_path, exist_ok=True)
    folder1_images = glob(os.path.join(folder1, '*'))
    folder2_images = glob(os.path.join(folder2, '*'))
    folder1_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder1_images}
    folder2_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder2_images}
    unmatched_folder1 = set(folder1_names.keys()) - set(folder2_names.keys())
    unmatched_folder2 = set(folder2_names.keys()) - set(folder1_names.keys())
    args_folder1 = [(name, folder1_names[name], unmatched_folder1_path) for name in unmatched_folder1]
    args_folder2 = [(name, folder2_names[name], unmatched_folder2_path) for name in unmatched_folder2]

    with Pool(cpu_count()) as pool:
        pool.map(move_unmatched_images, args_folder1)
        pool.map(move_unmatched_images, args_folder2)

    return len(unmatched_folder1), len(unmatched_folder2)
    
unmatched_folder1_path = 'unmatched1'  
unmatched_folder2_path = 'unmatched2'  
    
folder1_removed, folder2_removed = match_and_remove_images_by_name(
    r"/home/ml2/Desktop/Vscode/Background_removal/DIS/ISNET_2048/backdrop_testing/training/im",
    r"/home/ml2/Desktop/Vscode/Background_removal/DIS/ISNET_2048/backdrop_testing/training/gt",
    unmatched_folder1_path,
    unmatched_folder2_path
)   

print(f"Moved {folder1_removed} unmatched images from folder1 to {unmatched_folder1_path} and {folder2_removed} from folder2 to {unmatched_folder2_path}.")
