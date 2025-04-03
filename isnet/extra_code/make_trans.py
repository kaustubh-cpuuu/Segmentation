# from PIL import Image
# import cv2
# def apply_mask(base_image_path, mask_image_path, output_image_path):
#     # Open the base image
#
#
#     base_image = Image.fromarray(base_image_path)
#
#     # Open the mask image
#     mask_image = Image.fromarray(mask_image_path).convert("L")  # Convert to grayscale
#
#     # Resize the mask image to match the size of the base image
#     mask_image = mask_image.resize(base_image.size)
#
#     # Apply the mask as alpha channel
#     base_image.putalpha(mask_image)
#
#     # Save the result
#     base_image.save(output_image_path)
#
# #


from PIL import Image,ImageChops
import numpy as np
import cv2, time

def apply_mask(base_image_path, mask_image_path, output_image_path):
    # Open the base image
    base_image = Image.fromarray(base_image_path)
    # base_image = Image.open(base_image_path)
    # Open the mask image
    mask_image = Image.fromarray(mask_image_path).convert("L")  # Convert to grayscale

    # Resize the mask image to match the size of the base image
    mask_image = mask_image.resize(base_image.size )

    # Apply the mask as alpha channel
    base_image.putalpha(mask_image)
    image_array = np.array(base_image)

    # Invert the color channels (RGB)
    inverted_rgb = 255 - image_array[:, :, :3]

    # Merge the inverted RGB channels with the original alpha channel
    inverted_image_array = np.concatenate((inverted_rgb, image_array[:, :, 3:]), axis=2)

    return inverted_image_array


