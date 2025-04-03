import os
import shutil

def copy_image(image_name, input_folder, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct the full file path
    input_image_path = os.path.join(input_folder, image_name)

    # Check if the image exists in the input folder
    if os.path.exists(input_image_path):
        # Copy the image to the output folder
        shutil.copy(input_image_path, output_folder)
        print(f"Image '{image_name}' has been copied to '{output_folder}'")
    else:
        print(f"Image '{image_name}' not found in the input folder.")

# Example usage
input_folder = 'backdrop_cleaned/validation/im'
output_folder = 'backdrop_cleaned/validation/gtt'
image_name = 'backdrop_cleaned/training/imm/0Xed3KD8WPkYK7lZJfxomhifxlOwW4ZXntkqSF9toriWMXFtMWLPDTfDifg1pcMM20240920145329.jpg'  # Change this to the image name you want to copy

copy_image(image_name, input_folder, output_folder)