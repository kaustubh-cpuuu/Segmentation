# import os
# import cv2
# import imageio
# from PIL import Image, UnidentifiedImageError
# from multiprocessing import Pool, cpu_count

# # Paths to input and output folders
# input_folder_path = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/little_icon_lens/training/gt'
# output_folder_path = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/little_icon_lens/training/gt_png'

# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# def process_image(filename):
#     """Process a single image file."""
#     input_file_path = os.path.join(input_folder_path, filename)
    
#     # Skip if the file is a JPG
#     if filename.lower().endswith('.jpg'):
#         return f"Skipped JPG file: {filename}"

#     success = False
#     try:
#         # Try opening with PIL
#         img = Image.open(input_file_path)
#         img.load()
#         success = True
#     except (UnidentifiedImageError, IOError):
#         try:
#             # If PIL fails, try opening with OpenCV
#             cv_img = cv2.imread(input_file_path)
#             if cv_img is not None:
#                 cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(cv_img)
#                 success = True
#         except Exception:
#             pass  # Continue to the next attempt

#     if not success:
#         try:
#             # If OpenCV fails, try opening with imageio
#             io_img = imageio.imread(input_file_path)
#             img = Image.fromarray(io_img)
#             success = True
#         except Exception as e:
#             return f'Exception while processing {filename}: {e}'

#     # Convert and save the image in PNG format
#     new_filename = filename.rsplit('.', 1)[0] + '.png'
#     output_file_path = os.path.join(output_folder_path, new_filename)
#     try:
#         img.save(output_file_path, 'PNG')
#         return f'Converted {filename} to {new_filename}'
#     except Exception as e:
#         return f'Failed to save {new_filename}: {e}'

# def main():
#     # Get a list of all files in the input folder
#     filenames = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]

#     # Use multiprocessing Pool for parallel processing
#     with Pool(cpu_count()) as pool:
#         results = pool.map(process_image, filenames)

#     # Print results
#     for result in results:
#         print(result)

# if __name__ == '__main__':
#     main()






import os
import cv2
import imageio
from PIL import Image, UnidentifiedImageError
from multiprocessing import Pool, cpu_count

# Paths to input and output folders
input_folder_path = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/little_icon_lens/training/gt'
output_folder_path = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/little_icon_lens/training/gt_png'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def process_image(filename):
    """Process a single image file."""
    input_file_path = os.path.join(input_folder_path, filename)

    success = False
    try:
        # Try opening with PIL
        img = Image.open(input_file_path)
        img.load()
        success = True
    except (UnidentifiedImageError, IOError):
        try:
            # If PIL fails, try opening with OpenCV
            cv_img = cv2.imread(input_file_path)
            if cv_img is not None:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_img)
                success = True
        except Exception:
            pass  # Continue to the next attempt

    if not success:
        try:
            # If OpenCV fails, try opening with imageio
            io_img = imageio.imread(input_file_path)
            img = Image.fromarray(io_img)
            success = True
        except Exception as e:
            return f'Exception while processing {filename}: {e}'

    # Convert and save the image in PNG format
    new_filename = filename.rsplit('.', 1)[0] + '.png'
    output_file_path = os.path.join(output_folder_path, new_filename)
    try:
        img.save(output_file_path, 'PNG')
        return f'Converted {filename} to {new_filename}'
    except Exception as e:
        return f'Failed to save {new_filename}: {e}'

def main():
    # Get a list of all files in the input folder
    filenames = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]

    # Use multiprocessing Pool for parallel processing
    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, filenames)

    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
