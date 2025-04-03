# import cv2
# import numpy as np
# import os

# # Input directory containing "image" and "mask" folders
# input_directory = '/home/ml2/Desktop/Vscode/Background_removal/DIS/batch_5_6'
# output_mask_directory = os.path.join(input_directory, 'cluster_masks')
# output_image_directory = os.path.join(input_directory, 'cluster_images')

# # Create output directories if they don't exist
# os.makedirs(output_mask_directory, exist_ok=True)
# os.makedirs(output_image_directory, exist_ok=True)

# # Paths to "image" and "mask" folders
# image_folder = os.path.join(input_directory, 'im')
# mask_folder = os.path.join(input_directory, 'gt')

# # Get list of image and mask filenames
# image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
# mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

# # Process all images by pairing with names
# for image_file, mask_file in zip(image_files, mask_files):
#     # Load the corresponding image and mask
#     image_path = os.path.join(image_folder, image_file)
#     mask_path = os.path.join(mask_folder, mask_file)

#     image = cv2.imread(image_path)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#     # Apply corner detection (Harris method)
#     gray = np.float32(mask)

#     # Harris corner detection
#     dst = cv2.cornerHarris(gray, 25, 11, 0.04)

#     # Dilate the corner points to make them more visible
#     dst = cv2.dilate(dst, None)

#     # Threshold to get the corners as points
#     threshold = 0.01 * dst.max()
#     corners = dst > threshold

#     # Create a new blank grayscale image to store corners
#     corner_mask = np.zeros_like(mask, dtype=np.uint8)
#     corner_mask[corners] = 255  # Mark corners as white (255)

#     # Find contours to identify clusters of corners
#     contours, _ = cv2.findContours(corner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Prepare a list to store coordinates of expanded clusters
#     expanded_coordinates = []

#     for contour in contours:
#         # Calculate the bounding box for the contour
#         x, y, w, h = cv2.boundingRect(contour)

#         # Find the center of the bounding box
#         cx, cy = x + w // 2, y + h // 2

#         # Calculate the top-left corner of the expanded 64x64 box
#         expanded_x = max(cx - 32, 0)
#         expanded_y = max(cy - 32, 0)

#         # Calculate the bottom-right corner of the expanded 64x64 box
#         expanded_x2 = min(cx + 32, corner_mask.shape[1])
#         expanded_y2 = min(cy + 32, corner_mask.shape[0])

#         # Ensure the expanded box is exactly 64x64, adjusting if it exceeds image boundaries
#         if expanded_x2 - expanded_x < 64:
#             if expanded_x == 0:
#                 expanded_x2 = min(expanded_x + 64, corner_mask.shape[1])
#             else:
#                 expanded_x = max(expanded_x2 - 64, 0)

#         if expanded_y2 - expanded_y < 64:
#             if expanded_y == 0:
#                 expanded_y2 = min(expanded_y + 64, corner_mask.shape[0])
#             else:
#                 expanded_y = max(expanded_y2 - 64, 0)

#         # Save the coordinates of the expanded box
#         expanded_coordinates.append((expanded_x, expanded_y, expanded_x2, expanded_y2))

#         # Extract the 64x64 region from the mask and image
#         mask_crop = mask[expanded_y:expanded_y2, expanded_x:expanded_x2]
#         image_crop = image[expanded_y:expanded_y2, expanded_x:expanded_x2]

#         # Construct filenames based on coordinates
#         coordinates_str = f"{expanded_x}_{expanded_y}_{expanded_x2}_{expanded_y2}"
#         mask_filename = os.path.join(output_mask_directory, f"{os.path.splitext(mask_file)[0]}_{coordinates_str}.png")
#         image_filename = os.path.join(output_image_directory, f"{os.path.splitext(image_file)[0]}_{coordinates_str}.png")

#         # Save the crops
#         cv2.imwrite(mask_filename, mask_crop)
#         cv2.imwrite(image_filename, image_crop)

#     # Print the coordinates of the expanded clusters for the current file
#     for idx, (x1, y1, x2, y2) in enumerate(expanded_coordinates):
#         print(f"{mask_file} Cluster {idx + 1}: Top-Left ({x1}, {y1}), Bottom-Right ({x2}, {y2})")






import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count


def process_file(args):
    """
    Process a single image and mask pair.

    Args:
        args (tuple): Tuple containing image_file, mask_file, image_folder, mask_folder, output_image_directory, output_mask_directory.

    Returns:
        str: Log of the processed file.
    """
    try:
        image_file, mask_file, image_folder, mask_folder, output_image_directory, output_mask_directory = args

        # Load the corresponding image and mask
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load {image_path} or {mask_path}.")

        # Apply corner detection (Harris method)
        gray = np.float32(mask)
        dst = cv2.cornerHarris(gray, 25, 11, 0.04)
        dst = cv2.dilate(dst, None)

        # Threshold to get the corners as points
        threshold = 0.01 * dst.max()
        corners = dst > threshold

        # Create a blank image to store corners
        corner_mask = np.zeros_like(mask, dtype=np.uint8)
        corner_mask[corners] = 255

        # Find contours to identify clusters of corners
        contours, _ = cv2.findContours(corner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expanded_coordinates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            expanded_x = max(cx - 32, 0)
            expanded_y = max(cy - 32, 0)
            expanded_x2 = min(cx + 32, corner_mask.shape[1])
            expanded_y2 = min(cy + 32, corner_mask.shape[0])

            if expanded_x2 - expanded_x < 64:
                if expanded_x == 0:
                    expanded_x2 = min(expanded_x + 64, corner_mask.shape[1])
                else:
                    expanded_x = max(expanded_x2 - 64, 0)

            if expanded_y2 - expanded_y < 64:
                if expanded_y == 0:
                    expanded_y2 = min(expanded_y + 64, corner_mask.shape[0])
                else:
                    expanded_y = max(expanded_y2 - 64, 0)

            expanded_coordinates.append((expanded_x, expanded_y, expanded_x2, expanded_y2))

            # Extract the 64x64 region from the mask and image
            mask_crop = mask[expanded_y:expanded_y2, expanded_x:expanded_x2]
            image_crop = image[expanded_y:expanded_y2, expanded_x:expanded_x2]

            # Construct filenames based on coordinates
            coordinates_str = f"{expanded_x}_{expanded_y}_{expanded_x2}_{expanded_y2}"
            mask_filename = os.path.join(output_mask_directory, f"{os.path.splitext(mask_file)[0]}_{coordinates_str}.png")
            image_filename = os.path.join(output_image_directory, f"{os.path.splitext(image_file)[0]}_{coordinates_str}.png")

            # Save the crops
            cv2.imwrite(mask_filename, mask_crop)
            cv2.imwrite(image_filename, image_crop)

        # Log the processed file
        log = f"{mask_file} processed with {len(expanded_coordinates)} clusters."
        return log

    except Exception as e:
        return f"Error processing {image_file} and {mask_file}: {str(e)}"


def main():
    input_directory = '/home/ml2/Desktop/Vscode/Background_removal/DIS/batch_5_6'
    output_mask_directory = os.path.join(input_directory, 'cluster_masks')
    output_image_directory = os.path.join(input_directory, 'cluster_images')

    # Create output directories if they don't exist
    os.makedirs(output_mask_directory, exist_ok=True)
    os.makedirs(output_image_directory, exist_ok=True)

    # Paths to "image" and "mask" folders
    image_folder = os.path.join(input_directory, 'im')
    mask_folder = os.path.join(input_directory, 'gt')

    # Get list of image and mask filenames
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    # Ensure the number of images and masks match
    if len(image_files) != len(mask_files):
        print("Error: Number of images and masks do not match.")
        return

    # Prepare arguments for multiprocessing
    args = [
        (image_file, mask_file, image_folder, mask_folder, output_image_directory, output_mask_directory)
        for image_file, mask_file in zip(image_files, mask_files)
    ]

    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, args)

    # Print results
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
