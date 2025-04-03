
# import cv2
# import numpy as np
# import os

# def load_image(image_path):
#     """Load an image from a file path."""
#     return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# def apply_canny_edge_detection(image, threshold1, threshold2):
#     """Apply Canny edge detection to an image."""
#     return cv2.Canny(image, threshold1, threshold2)


# def draw_wide_edges_on_image(original_image, edges, width):
#     """Draw a wide path around the edges on the original image with grey color."""
#     result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
#     # Dilate the edges to create a wider path
#     kernel = np.ones((width, width), np.uint8)
#     wide_edges = cv2.dilate(edges, kernel, iterations=1)
#     result_image[wide_edges != 0] = [128, 128, 128]  # Drawing edges in grey
#     return result_image


# def save_image(image, output_path):
#     """Save the image to a file."""
#     cv2.imwrite(output_path, image)


# def process_image(image_path, threshold1, threshold2, output_dir, width):
#     image = load_image(image_path)
#     if image is None:
#         print(f"Error: Unable to load image from path: {image_path}")
#         return

#     edges = apply_canny_edge_detection(image, threshold1, threshold2)
#     result_image = draw_wide_edges_on_image(image, edges, width)

#     filename = os.path.basename(image_path)
#     output_path = os.path.join(output_dir, filename)
#     save_image(result_image, output_path)
#     print(f"Result image saved to {output_path}")


# def main():
#     # Define paths and parameters directly in the code
#     image_path = "/home/ml2/Desktop/Vscode/sch_bg_isnet/ormbg/demo_dataset/GT"  # Path to the image file or folder
#     threshold1 = 10  # First threshold for Canny edge detection
#     threshold2 = 10  # Second threshold for Canny edge detection
#     output_dir = "/home/ml2/Desktop/Vscode/sch_bg_isnet/ormbg/demo_dataset/out"  # Directory to save the result images
#     width = 5  # Width of the path around the edges

#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     if os.path.isdir(image_path):
#         # Process each image in the folder
#         for filename in os.listdir(image_path):
#             file_path = os.path.join(image_path, filename)
#             if os.path.isfile(file_path):
#                 process_image(file_path, threshold1, threshold2, output_dir, width)
#     else:
#         # Process a single image
#         process_image(image_path, threshold1, threshold2, output_dir, width)


# if __name__ == "__main__":
#     main()




import cv2
import numpy as np
import os

class Trimap:
    def __init__(self, threshold1, threshold2, width, output_dir):
        """
        Initialize the Trimap class.
        :param threshold1: First threshold for Canny edge detection.
        :param threshold2: Second threshold for Canny edge detection.
        :param width: Width of the path around the edges.
        :param output_dir: Directory to save output images.
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.width = width
        self.output_dir = output_dir
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_image(self, image_path):
        """Load an image from the given path."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to load image from path: {image_path}")
        return image

    def apply_canny_edge_detection(self, image):
        """Apply Canny edge detection to the given image."""
        return cv2.Canny(image, self.threshold1, self.threshold2)

    def draw_wide_edges_on_image(self, original_image, edges):
        """Draw a wide path around edges and create a grey overlay."""
        result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        # Dilate the edges to widen them
        kernel = np.ones((self.width, self.width), np.uint8)
        wide_edges = cv2.dilate(edges, kernel, iterations=1)
        result_image[wide_edges != 0] = [128, 128, 128]  # Drawing edges in grey
        return result_image

    def save_image(self, image, filename):
        """Save the resulting image to the output directory."""
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Result image saved to {output_path}")

    def process_image(self, image_path):
        """Process a single image: load, apply edge detection, draw edges, and save."""
        try:
            image = self.load_image(image_path)
            edges = self.apply_canny_edge_detection(image)
            result_image = self.draw_wide_edges_on_image(image, edges)
            filename = os.path.basename(image_path)
            self.save_image(result_image, filename)
        except Exception as e:
            print(e)

    def process_folder(self, folder_path):
        """Process all images in a folder."""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                print(f"Processing: {file_path}")
                self.process_image(file_path)

    


def main():
    # Define parameters
    image_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/Input_1_image"  # Path to image or folder
    output_dir = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/Input_1_image/trial"  # Directory for output
    threshold1 = 20  # Canny threshold 1
    threshold2 = 20  # Canny threshold 2
    width = 10  # Width for dilating edges

    # Create a Trimap object
    trimap_processor = Trimap(threshold1, threshold2, width, output_dir)

    # Process folder or single image
    if os.path.isdir(image_path):
        trimap_processor.process_folder(image_path)
    else:
        trimap_processor.process_image(image_path)


if __name__ == "__main__":
    main()

