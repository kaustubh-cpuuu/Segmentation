import os   
import cv2
import numpy as np
from multiprocessing import Pool

class ImageMaskProcessor:
    def __init__(self, image_folder, mask_folder, output_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def process_image_mask_pair(self, args):
        """Process a single image-mask pair."""
        image_path, mask_path, output_folder = args

        # Load the image (JPG) and mask (PNG)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error: Could not load {image_path} or {mask_path}")
            return

        # Ensure the mask dimensions match the image
        if image.shape[:2] != mask.shape[:2]:
            print(f"Error: Dimensions do not match for {image_path} and {mask_path}")
            return

        # Create a green overlay based on the mask
        green_overlay = np.zeros_like(image)
        green_overlay[:, :, 2] = mask  # Assign mask to the green channel

        # Add the green overlay with 50% opacity
        overlay = cv2.addWeighted(image, 1.0, green_overlay, 0.9, 0)

        # Save the result
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, overlay)
        print(f"Processed: {image_path} -> {output_path}")

    def process_folder(self):
        """Process all image-mask pairs in the folder using multiprocessing."""
        # Collect image-mask pairs
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith('.jpg')]
        tasks = []

        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            mask_path = os.path.join(self.mask_folder, image_file.replace('.jpg', '.png'))  # Expect PNG mask

            if os.path.exists(mask_path):
                tasks.append((image_path, mask_path, self.output_folder))
            else:
                print(f"Warning: Mask not found for {image_file}")

        # Process tasks with multiprocessing
        with Pool() as pool:
            pool.map(self.process_image_mask_pair, tasks)

# Example usage
if __name__ == "__main__":
    image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/im"       # Replace with your image folder path
    mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/gt"         # Replace with your mask folder path
    output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/color2"      # Replace with your desired output folder path

    processor = ImageMaskProcessor(image_folder, mask_folder, output_folder)
    processor.process_folder()
