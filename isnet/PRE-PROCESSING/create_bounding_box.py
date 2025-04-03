import os
from PIL import Image
import numpy as np


# Function to get bounding box coordinates for a specific color
# def get_bounding_box_coords(image_array, color):
#     rows, cols = np.where(np.all(image_array[:, :, :3] == color, axis=2))
#     if rows.size and cols.size:
#         x_min, x_max = cols.min(), cols.max()
#         y_min, y_max = rows.min(), rows.max()
#         return x_min, y_min, x_max, y_max
#     else:
#         return None

# function to get bounding box co-ordinate for a specific color


def get_bounding_box_coords(image_array, color, expand_pixels=0):
    rows, cols = np.where(np.all(image_array[:, :, :3] == color, axis=2))
    if rows.size and cols.size:
        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()

        x_min = max(0, x_min - expand_pixels)
        y_min = max(0, y_min - expand_pixels)
        x_max = min(image_array.shape[1] - 1, x_max + expand_pixels)
        y_max = min(image_array.shape[0] - 1, y_max + expand_pixels)

        return x_min, y_min, x_max, y_max
    else:
        return None

# Function to normalize the bounding box coordinates
def convert_to_normalized_format(coords, img_width, img_height, class_id):
    x_min, y_min, x_max, y_max = coords
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    normalized_center_x = center_x / img_width
    normalized_center_y = center_y / img_height
    normalized_width = width / img_width
    normalized_height = height / img_height
    return class_id, normalized_center_x, normalized_center_y, normalized_width, normalized_height


# Process images and save bounding boxes
def process_images(input_dir, output_dir, color_labels, class_ids):
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, file_name)
            image = Image.open(image_path)
            image_np = np.array(image)
            image_width, image_height = image.size

            bounding_boxes = {}
            for color, label in color_labels.items():
                coords = get_bounding_box_coords(image_np, color)
                if coords:
                    class_id = class_ids[label]
                    bounding_boxes[label] = convert_to_normalized_format(
                        coords, image_width, image_height, class_id)

            # Exclude the yellow color bounding box
            bounding_boxes.pop('shock', None)
            # bounding_boxes.pop('skin', None)
            # bounding_boxes.pop('spects', None)
            # bounding_boxes.pop('cap', None)
            # bounding_boxes.pop('Red', None)
            # bounding_boxes.pop('light_b', None)

            # Prepare the content in the requested format
            normalized_content = [
                f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n"
                for class_id, norm_center_x, norm_center_y, norm_width, norm_height in bounding_boxes.values()
            ]

            # Create the output directory if it does not exist
            os.makedirs(output_dir, exist_ok=True)

            # Define the output file path
            base_name = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(output_dir, f"{base_name}.txt")

            # Write the normalized content to the text file
            with open(output_file_path, 'w') as file:
                file.writelines(normalized_content)

# Define the input and output directories
input_dir = 'output_mask/T-shirt_red_pink_5000' # Update this to your input directory
output_dir = 'output_mask/T-shirt_red_pink_5000_labels'  # Update this to your output directory


# color_labels = {
#     (255, 0, 0): 'tshirt',
#     (0, 119, 221): 'Jacket',
#     (0, 0, 85): 'Top',
#     (0, 255, 0): 'Bottom',
#     (0, 216, 255): 'shoes',
#     (255, 85, 0): 'Shirt',
#     (0 , 128 , 0) : 'Bag' ,
#     (255 , 228 , 0 ) : 'skin',
#     (163 , 198 , 0) : 'spects' ,
#     (10 , 61 , 0) : 'cap'
#
# }
#
#
# class_ids = {
#
#     'tshirt': 0,
#     "Jacket": 0,
#     'Top': 0,
#     "Bottom": 1,
#     "shoes": 2,
#     "Shirt": 0,
#     "Bag": 6,
#     "skin": 7,
#     "spects": 8,
#     "cap": 9
#
# }


color_labels = {
    (255, 165, 0): 'jacket',
    (255,192,203): 'shirt',
    (255,255,0): 'onepiece',
    (0, 128, 0): 'Bottom',
    (255, 0, 0): 'T-shirt',
    (0, 0, 255): 'shoes',
    (255 , 255 , 255) : 'shock',


}


class_ids = {

    'jacket': 0,
    "shirt": 0,
    'onepiece': 5,
    "Bottom": 1,
    "T-shirt": 0,
    "shoes": 2,
    "shock": 6,

}


# Run the processing function
process_images(input_dir, output_dir, color_labels, class_ids)
