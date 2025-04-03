







#Without multiprocesing

# import cv2
# import cv2
# import numpy as np
# import os
# import time


# def gamma_correction(image_path, output_path, gamma=5.2):
#     try:
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             return
#         image = image / 255.0
#         adjust_image = np.power(image, gamma)
#         adjust_image = (adjust_image * 255.0).astype(np.uint8)

#         cv2.imwrite(output_path, adjust_image)
#         print(f"Gamma correction applied and saved: {output_path}")

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")


# def photocopy_filter(image_path, output_path, detail=353, gamma=2.4):
#     try:

#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             return

#         image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         inverted_image = cv2.bitwise_not(gray_image)
#         detail = detail if detail % 2 == 1 else detail + 1  # Ensure odd kernel size
#         blurred_image = cv2.GaussianBlur(inverted_image, (detail, detail), 0)
#         photocopy_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
#         photocopy_image = cv2.adaptiveThreshold(
#             photocopy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 2
#         )

#         cv2.imwrite(output_path, photocopy_image)
#         print(f"Photocopy filter applied and saved: {output_path}")

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")


# def process_images_combined(input_folder, temp_folder, output_folder, detail=353, gamma_correction_value=5.2, filter_gamma=2.4):
#     os.makedirs(temp_folder, exist_ok=True)
#     os.makedirs(output_folder, exist_ok=True)

#     image_paths = [
#         os.path.join(input_folder, filename)
#         for filename in os.listdir(input_folder)
#         if filename.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
#     for image_path in image_paths:
#         temp_path = os.path.join(temp_folder, os.path.basename(image_path))
#         gamma_correction(image_path, temp_path, gamma=gamma_correction_value)

#     for temp_file in os.listdir(temp_folder):
#         temp_path = os.path.join(temp_folder, temp_file)
#         final_path = os.path.join(output_folder, temp_file)
#         photocopy_filter(temp_path, final_path, detail=detail, gamma=filter_gamma)


# if __name__ == "__main__":

#     input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_input"
#     temp_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_output"
#     output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_filter"

#     # Time the process
#     start_time = time.time()
#     process_images_combined(input_folder, temp_folder, output_folder)
#     print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
 





# import cv2
# import numpy as np
# import os
# import time

# def gamma_correction(image, gamma=5.2):

#     try:
#         lookup_table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype(np.uint8)
#         return cv2.LUT(image, lookup_table)
#     except Exception as e:
#         print(f"Error in gamma correction: {e}")
#         return None

# def photocopy_filter(image, detail=353, gamma=2.4):

#     try:
#         image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         inverted_image = cv2.bitwise_not(gray_image)
#         detail = max(3, detail if detail % 2 == 1 else detail + 1)  # Ensure odd kernel size
#         blurred_image = cv2.GaussianBlur(inverted_image, (detail, detail), 0)

#         photocopy_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
#         photocopy_image = cv2.adaptiveThreshold(
#             photocopy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 2
#         )
#         return photocopy_image
#     except Exception as e:
#         print(f"Error in photocopy filter: {e}")
#         return None

# def process_images_combined(input_folder, output_folder, detail=353, gamma_correction_value=5.2, filter_gamma=2.4):

#     try:
#         os.makedirs(output_folder, exist_ok=True)

#         image_paths = [
#             os.path.join(input_folder, filename)
#             for filename in os.listdir(input_folder)
#             if filename.lower().endswith((".jpg", ".jpeg", ".png"))
#         ]

#         if not image_paths:
#             print("No valid images found in the input folder.")
#             return

#         for image_path in image_paths:
#             try:
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     print(f"Failed to load image: {image_path}")
#                     continue

#                 adjusted_image = gamma_correction(image, gamma=gamma_correction_value)
#                 if adjusted_image is None:
#                     continue

#                 final_image = photocopy_filter(adjusted_image, detail=detail, gamma=filter_gamma)
#                 if final_image is None:
#                     continue
#                 output_path = os.path.join(output_folder, os.path.basename(image_path))
#                 cv2.imwrite(output_path, final_image)
#                 print(f"Processed and saved: {output_path}")

#             except Exception as e:
#                 print(f"Error processing {image_path}: {e}")
#     except Exception as e:
#         print(f"Error in processing images: {e}")

# if __name__ == "__main__":
#     input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_input"
#     output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_filter"

#     start_time = time.time()
#     process_images_combined(input_folder, output_folder)
#     print(f"Processing completed in {time.time() - start_time:.2f} seconds.")





import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import time

def gamma_correction(image_path_and_output_path):
    image_path, output_path = image_path_and_output_path
    try:

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        gamma = 5.2
        adjust_image = np.power(image, gamma)
        adjust_image = (adjust_image * 255.0).astype(np.uint8)

        cv2.imwrite(output_path, cv2.cvtColor(adjust_image, cv2.COLOR_RGB2BGR))
        print(f"Level adjusted and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def photocopy_filter(args):
    image_path, output_path, detail, darkness, gamma = args
    try:

        image = cv2.imread(image_path)
        image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray_image)
        detail = detail if detail % 2 == 1 else detail + 1  
        blurred_image = cv2.GaussianBlur(inverted_image, (detail, detail), 0)
        photocopy_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
        photocopy_image = cv2.adaptiveThreshold(
            photocopy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 2
        )

        # Save the resulting image
        cv2.imwrite(output_path, photocopy_image)
        print(f"Photocopy filter applied and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images_combined(input_folder, temp_folder, output_folder, detail=353, darkness=200, gamma=2.4):
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    temp_paths = [
        os.path.join(temp_folder, os.path.basename(image_path))
        for image_path in image_paths
    ]


    with Pool(processes=cpu_count()) as pool:
        pool.map(gamma_correction, zip(image_paths, temp_paths))
    final_paths = [
        os.path.join(output_folder, os.path.basename(temp_path))
        for temp_path in temp_paths
    ]
    tasks = [
        (temp_path, final_path, detail, darkness, gamma)
        for temp_path, final_path in zip(temp_paths, final_paths)
    ]
    with Pool(processes=cpu_count()) as pool:
        pool.map(photocopy_filter, tasks)




if __name__ == "__main__":
    input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_input"  # Input folder
    temp_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_output"  # Temporary folder for level adjustment
    output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_filter"  # Final output folder

    process_images_combined(input_folder, temp_folder, output_folder)

