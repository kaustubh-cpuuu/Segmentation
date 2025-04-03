from PIL import Image
import os
import numpy as np

def resize_images(folder_name, size, after_image, before_image, output_folder):
    fldr = output_folder + str(size)
    if not os.path.exists(fldr):
        os.makedirs(fldr)
    if not os.path.exists(fldr + "/after/"):
        os.makedirs(fldr + "/after/")
    if not os.path.exists(fldr + "/before/"):
        os.makedirs(fldr + "/before/")
    resized_after_img = after_image.resize((size, size), Image.NEAREST)
    resized_after_img.save(
        fldr + "/after/" + folder_name + ".png",
        "PNG",
    )
    resized_before_img = before_image.resize((size, size), Image.NEAREST)
    resized_before_img.save(
        fldr + "/before/" + folder_name + ".png",
        "PNG",
    )


def process_all_folders(parent_folder_path, output_folder):
    for folder_name in os.listdir(parent_folder_path):
        potential_folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(potential_folder_path):
            for sub_folder in ["after", "before"]:
                sub_folder_path = os.path.join(potential_folder_path, sub_folder)
                if os.path.exists(sub_folder_path) and os.path.isdir(sub_folder_path):
                    for file in os.listdir(sub_folder_path):
                        if file.endswith(".png") and "mask" in file:
                            file_path = os.path.join(sub_folder_path, file)
                            img = Image.open(file_path).convert("L")
                            img = img.point(lambda p: 255 if p != 0 else 0)
                            img.save(file_path)


def remove_anti_aliasing_for_mask(img, threshold=128):
    """
    Remove anti-aliasing from a binary mask image.
    Assumes the mask is white (255, 255, 255) with transparent background.
    """
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[3] < threshold:  # Check transparency channel
            newData.append((255, 255, 255, 0))
        else:
            newData.append((255, 255, 255, 255))
    img.putdata(newData)
    return img


def process_images_in_folder(folder_path, output_folder, process_function):
    """Processes all .png files in the given folder using the provided processing function."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            try:
                file_path = os.path.join(folder_path, file)
                img = Image.open(file_path).convert("RGBA")
                processed_img = process_function(img)

                save_path = os.path.join(output_folder, file)
                processed_img.save(save_path)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Provide the path to the folder containing PNG images
before_folder_path = "product/after"
after_folder_path = "product/before"
output_folder = "res"

# Process images in 'before' folder
process_images_in_folder(before_folder_path, output_folder, remove_anti_aliasing_for_mask)
