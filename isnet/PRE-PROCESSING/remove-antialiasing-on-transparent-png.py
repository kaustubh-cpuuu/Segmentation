# import os
# from PIL import Image

# def remove_anti_aliasing_transparent_bg(image_path, save_path, threshold=128, output_size=None):
#     # Load the image
#     image = Image.open(image_path).convert("RGBA")  # Work with RGBA to keep transparency

#     # If output_size isn't specified, use the original image's size
#     if output_size is None:
#         output_size = image.size

#     # Split into RGB and A channels
#     r, g, b, alpha = image.split()

#     # Convert RGB channels to grayscale (use any one since they're now identical)
#     grayscale = r.convert("L")

    
#     # Apply threshold to create black-and-white image
#     bw = grayscale.point(lambda p: 255 if p > threshold else 0)

#     # Prepare to merge: make the original transparent areas (alpha=0) black in bw
#     black_bg = Image.new("L", image.size, "black")

#     # Merge the black-and-white image with black background based on transparency
#     rgb = Image.merge("RGB", [bw, bw, bw])
#     final_image = Image.composite(rgb, Image.merge("RGB", [black_bg, black_bg, black_bg]), alpha)

#     # Resize the image
#     final_image = final_image.resize(output_size, Image.Resampling.NEAREST)

#     # Save the processed image
#     final_image.save(save_path, "PNG")  # Ensure saving in PNG to keep transparency handling correct

#     return save_path


# def process_folder(input_folder, output_folder, threshold=128):

#     # Ensure output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # List all files in the input folder
#     for file_name in os.listdir(input_folder):
#         # Construct full file path
#         input_path = os.path.join(input_folder, file_name)
#         output_path = os.path.join(output_folder, file_name)

#         # Check if the file is an image (by a simple extension check, for demonstration)
#         if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             try:
#                 # Process and save the image
#                 remove_anti_aliasing_transparent_bg(input_path, output_path, threshold)
#                 print(f"Processed and saved: {output_path}")
#             except Exception as e:
#                 print(f"Failed to process {input_path}: {e}")


# # Example usage
# input_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt"
# output_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt_anti"
# process_folder(input_folder, output_folder, threshold=64)


import os
import multiprocessing
from PIL import Image

def remove_anti_aliasing_transparent_bg(image_path, save_path, threshold=128, output_size=None):
    try:
        image = Image.open(image_path).convert("RGBA")
        if output_size is None:
            output_size = image.size
        r, g, b, alpha = image.split()
        grayscale = r.convert("L")
        bw = grayscale.point(lambda p: 255 if p > threshold else 0)
        black_bg = Image.new("L", image.size, "black")
        rgb = Image.merge("RGB", [bw, bw, bw])
        final_image = Image.composite(rgb, Image.merge("RGB", [black_bg, black_bg, black_bg]), alpha)
        final_image = final_image.resize(output_size, Image.Resampling.NEAREST)
        final_image.save(save_path, "PNG")
        print(f"Processed: {save_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_image(args):
    input_path, output_path, threshold = args
    remove_anti_aliasing_transparent_bg(input_path, output_path, threshold)

def process_folder(input_folder, output_folder, threshold=128, num_workers=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    args_list = [(os.path.join(input_folder, file_name), os.path.join(output_folder, file_name), threshold) for file_name in image_files]
    
    num_workers = num_workers or multiprocessing.cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_image, args_list)

if __name__ == "__main__":
    input_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt"
    output_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/Backdrop/training/gt_anti"
    process_folder(input_folder, output_folder, threshold=64)
