import os
import shutil

def move_image_and_mask(image_name, image_folder, mask_folder, output_image_folder, output_mask_folder):
    # Ensure output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # Search for the image in the image folder
    image_path = os.path.join(image_folder, image_name)
    if not os.path.exists(image_path):
        print(f"Image '{image_name}' not found in the image folder.")
        return

    # Get the base name (without extension) to match the mask
    image_base_name = os.path.splitext(image_name)[0]

    # Search for the corresponding mask in the mask folder
    mask_path = None
    for file in os.listdir(mask_folder):
        if os.path.splitext(file)[0] == image_base_name:
            mask_path = os.path.join(mask_folder, file)
            break

    if not mask_path:
        print(f"No corresponding mask found for image '{image_name}'.")
        return

    # Move the image to the output image folder
    shutil.copy(image_path, os.path.join(output_image_folder, image_name))
    print(f"Moved image '{image_name}' to '{output_image_folder}'.")

    # Move the mask to the output mask folder
    shutil.copy(mask_path, os.path.join(output_mask_folder, os.path.basename(mask_path)))
    print(f"Moved mask '{os.path.basename(mask_path)}' to '{output_mask_folder}'.")


# Example usage
image_folder = "bg_removal_all_in_one/validation/im"
mask_folder = "bg_removal_all_in_one/validation/gt"
output_image_folder = "bg_removal_all_in_one/training/im_error"
output_mask_folder = "bg_removal_all_in_one/training/gt_error"
image_name = "1bhqz5BylWGV30omg7WX2NPqCxyKnZgXP1aK2RDNvPVT3dABHPu297l0oST9fjh820240928141950g.jpg"  # Replace with the image name you want to search for

move_image_and_mask(image_name, image_folder, mask_folder, output_image_folder, output_mask_folder)
