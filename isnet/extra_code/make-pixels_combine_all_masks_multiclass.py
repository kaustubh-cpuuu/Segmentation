import csv
import os
import numpy as np
from PIL import Image
import random
import sys


csv.field_size_limit(2147483647)

# Define the set of ClassIds to avoid
avoided_classes = {
    # glasses, hat, headband, head covering, hair accessory, tie, glove, watch, belt, leg warmer, tights, stockings, bag, wallet, scarf, umbrella starts from here
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    24,
    25,
    26,
    # extra object classes
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
}


# avoided_classes = {
#     13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26,
#     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
#     39, 40, 41, 42, 43, 44, 45
# }


def get_hex_code(color_name):
    colors = {
        "orange": (255, 165, 0),
        "red": (255, 0, 0),
        "pink": (255, 192, 203),
        "green": (0, 128, 0),
        "yellow": (255, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }
    return colors.get(color_name, (255, 255, 255))  # Default to white if not found


def get_fill_color(class_id):
    # Define colors for specific classes
    if class_id in (1, 2):
        return get_hex_code("red")
    if class_id == 0:
        return get_hex_code("pink")
    if class_id in (4, 5, 9, 12, 3):
        return get_hex_code("orange")
    if class_id in (6, 7, 8):
        return get_hex_code("green")
    if class_id in (10, 11):
        return get_hex_code("yellow")
    if class_id == 23:
        return get_hex_code("blue")
    return get_hex_code("white")


def rle_decode(encoded_pixels, height, width):
    """
    Decode RLE-encoded pixels into a binary mask.
    """
    shape = (width, height)
    s = encoded_pixels.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def combine_masks(masks, colors, height, width):
    """
    Combine multiple RGB masks into a single RGB mask.
    """
    combined_mask = np.zeros((height, width, 3), dtype=np.uint8)  # RGB
    for mask, color in zip(masks, colors):
        for i in range(3):  # Apply color to each channel
            combined_mask[:, :, i] = np.where(
                mask > 0, color[i], combined_mask[:, :, i]
            )
    return combined_mask


def save_combined_mask(folder_path, image_id, combined_mask):
    """
    Save the combined RGB mask as an image.
    """
    mask_image = Image.fromarray(combined_mask, "RGB")
    save_image_name = os.path.splitext(image_id)[0] + ".png"
    file_path = os.path.join(folder_path, save_image_name)
    mask_image.save(file_path)


csv_file_path = (
    r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\csv-divided-in-15-parts-for-easy-loading\mini_15.csv"
)
folder_path = (
    r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\train"
)


masks_dict = {}

# # Updated part of the code to include dimensions in masks_dict
# with open(csv_file_path, newline="") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         class_id = row["ClassId"]
#         if class_id not in avoided_classes:
#             image_id = row["ImageId"]
#             encoded_pixels = row["EncodedPixels"]
#             height = int(row["Height"])
#             width = int(row["Width"])

#             if encoded_pixels:
#                 mask = rle_decode(encoded_pixels, height, width)
#                 # When adding masks to the dictionary, also store class IDs
#                 if image_id in masks_dict:
#                     masks_dict[image_id]["masks"].append(mask)
#                     masks_dict[image_id]["class_ids"].append(
#                         class_id
#                     )  # Store class IDs
#                 else:
#                     masks_dict[image_id] = {
#                         "masks": [mask],
#                         "class_ids": [class_id],  # Initialize with the first class ID
#                         "dimensions": (height, width),
#                     }

with open(csv_file_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Process ClassId to use the part before any underscore
        class_id = row["ClassId"]
        class_id = class_id.split("_")[0]  # Keep only the part before the underscore
        class_id = int(class_id)  # Convert to integer if needed for comparison

        if class_id not in avoided_classes:
            image_id = row["ImageId"]
            encoded_pixels = row["EncodedPixels"]
            height = int(row["Height"])
            width = int(row["Width"])

            if encoded_pixels:
                mask = rle_decode(encoded_pixels, height, width)
                # When adding masks to the dictionary, also store class IDs
                if image_id in masks_dict:
                    masks_dict[image_id]["masks"].append(mask)
                    masks_dict[image_id]["class_ids"].append(
                        class_id
                    )  # Store class IDs
                else:
                    masks_dict[image_id] = {
                        "masks": [mask],
                        "class_ids": [class_id],  # Initialize with the first class ID
                        "dimensions": (height, width),
                    }


output_folder_path=r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\output_mask"
full_folder_path = os.path.join(output_folder_path, "combined_masks_multiclass4")
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)

for index, (image_id, data) in enumerate(masks_dict.items()):
    masks = data["masks"]
    class_ids = data["class_ids"]  # Now this line should work as class_ids are stored
    colors = [
        get_fill_color(class_id) for class_id in class_ids
    ]  # Determine colors for each mask
    height, width = data["dimensions"]
    combined_mask = combine_masks(
        masks, colors, height, width
    )  # Correctly call combine_masks with all arguments
    save_combined_mask(
        full_folder_path, image_id, combined_mask
    )  # Save the combined RGB mask
    print(f"Processing {index + 1}/{len(masks_dict)}: {image_id}")
