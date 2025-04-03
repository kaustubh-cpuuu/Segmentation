import csv
import os
import numpy as np
from PIL import Image
import random

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
    # if class_id in (1, 2):
    #     return get_hex_code("red")
    # if class_id == 0:
    #     return get_hex_code("pink")
    # if class_id in (4, 5, 9, 12, 3):
    #     return get_hex_code("orange")
    # if class_id in (6, 7, 8):
    #     return get_hex_code("green")
    # if class_id in (10, 11):
    #     return get_hex_code("yellow")
    # if class_id == 23:
    #     return get_hex_code("blue")
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


def save_image_from_encoded_pixels(
    folder_path, image_id, encoded_pixels, height, width, image_count, class_id
):
    mask = rle_decode(encoded_pixels, height, width)

    color = get_fill_color(class_id)  # Get color based on class_id
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)  # Using RGB
    colored_mask[:, :] = [0, 0, 0]  # Set the background to black

    for i in range(3):
        colored_mask[:, :, i] = np.where(mask > 0, color[i], colored_mask[:, :, i])

    mask_image = Image.fromarray(colored_mask, "RGB")

    save_image_name = os.path.splitext(image_id)[0]
    file_path = os.path.join(folder_path, f"{save_image_name}.png")
    mask_image.save(file_path)
    return file_path


def combine_masks(masks, height, width):
    """
    Combine multiple masks into a single mask with a specified background and mask color.
    """
    combined_mask = np.zeros(
        (height, width), dtype=np.uint8
    )  # Start with a black background
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)  # Combine masks
    return combined_mask


def save_combined_mask(folder_path, image_id, combined_mask):
    """
    Save the combined mask as an image with a white mask on a black background.
    """
    height, width = combined_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)  # RGB
    color = get_hex_code("white")  # White color for the mask
    for i in range(3):
        colored_mask[:, :, i] = np.where(
            combined_mask > 0, color[i], 0
        )  # Apply color to the mask

    mask_image = Image.fromarray(colored_mask, "RGB")
    save_image_name = os.path.splitext(image_id)[0] + ".png"
    file_path = os.path.join(folder_path, save_image_name)
    mask_image.save(file_path)


csv_file_path = (
    "D:/data-cleanup/iMaterialist/csv-divided-in-15-parts-for-easy-loading/mini_1.csv"
)
folder_path = (
    "D:/data-cleanup/iMaterialist/csv-divided-in-15-parts-for-easy-loading/mini_1"
)

masks_dict = {}

# Updated part of the code to include dimensions in masks_dict
with open(csv_file_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_id = int(row["ClassId"])
        if class_id not in avoided_classes:
            image_id = row["ImageId"]
            encoded_pixels = row["EncodedPixels"]
            height = int(row["Height"])
            width = int(row["Width"])

            if encoded_pixels:
                mask = rle_decode(encoded_pixels, height, width)
                if image_id in masks_dict:
                    masks_dict[image_id]["masks"].append(mask)
                else:
                    masks_dict[image_id] = {
                        "masks": [mask],
                        "dimensions": (height, width),
                    }


full_folder_path = os.path.join(folder_path, "combined_masks")
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)

# Updated loop to use the correct dimensions for each image
for index, (image_id, data) in enumerate(masks_dict.items()):
    masks = data["masks"]
    height, width = data["dimensions"]  # Retrieve the correct dimensions
    combined_mask = combine_masks(masks, height, width)  # Use the correct dimensions
    save_combined_mask(full_folder_path, image_id, combined_mask)
    print(f"Processing {index + 1}/{len(masks_dict)}: {image_id}")
