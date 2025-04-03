import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet_DenseNet_121 import ISNetDIS
import time

def crop_and_save_image(image, box, expansion_pixels, destination_path):
    """
    Crop the image based on the bounding box with expansion and save it to the specified path.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]  # Convert box coordinates to integer

    # Expand the bounding box by the specified number of pixels
    x1 = max(0, x1 - expansion_pixels)
    y1 = max(0, y1 - expansion_pixels)
    x2 = min(image.shape[1], x2 + expansion_pixels)
    y2 = min(image.shape[0], y2 + expansion_pixels)

    cropped_image = image[y1:y2, x1:x2]  # Crop image
    if cropped_image.size == 0:
        print("Empty cropped image, skipping.")
        return False
    cv2.imwrite(destination_path, cropped_image)  # Save cropped image
    return True

def process_images_yolo(source, destination, yolo_model, isnet_model, names):
    if not os.path.exists(source):
        print(f"Source directory {source} does not exist.")
        return

    if not os.path.exists(destination):
        os.makedirs(destination)

    folders_list = os.listdir(source)
    for folder_name in folders_list:
        folder_path = os.path.join(source, folder_name)
        if os.path.isdir(folder_path):
            process_folder_yolo(folder_path, destination, yolo_model, isnet_model, names)
        else:
            process_folder_yolo(source, destination, yolo_model, isnet_model, names)
            break  # Processing is complete

def save_classified_image_yolo(image, destination, boxes, isnet_model):
    """
    Save the cropped T-shirt images.
    """
    saved = False
    for i, box in enumerate(boxes):  # Assuming multiple T-shirts could be present
        expansion_pixels = 20
        cropped_image_path = f"temp_tshirt_{i}.png"
        cropped_image = crop_and_save_image(image, box, expansion_pixels, cropped_image_path)
        if cropped_image:
            print("Processing cropped image:", cropped_image_path)  # Add this line for debugging

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = [int(coord) for coord in box]

            """saved results """
            isnet_result = process_image_isnet(cropped_image_path, isnet_model)  # Pass image path directly
            result_path = os.path.join(destination, f"result_tshirt_{i}.png")

            # Calculate expanded dimensions of the cropped part
            expanded_x1 = max(0, x1 - expansion_pixels)
            expanded_y1 = max(0, y1 - expansion_pixels)
            expanded_x2 = min(image.shape[1], x2 + expansion_pixels)
            expanded_y2 = min(image.shape[0], y2 + expansion_pixels)
            expanded_width = expanded_x2 - expanded_x1
            expanded_height = expanded_y2 - expanded_y1

            # Resize ISNet result to match expanded dimensions
            isnet_result_resized = cv2.resize(isnet_result, (expanded_width, expanded_height))

            # Create black background image with the same dimensions as the original image
            black_image = np.zeros_like(image)

            # Paste resized ISNet result into the expanded cropped part of the black image
            black_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = isnet_result_resized

            # Save modified black image as output
            cv2.imwrite(result_path, black_image)

            """Saved cropped image"""
            original_cropped_path = os.path.join(destination, f"original_cropped_tshirt_{i}.png")
            cv2.imwrite(original_cropped_path, cv2.imread(cropped_image_path))

            saved = True
    return saved



def process_image_isnet(image, model):
    input_size = [1024, 1024]
    print("Processing image:", image)  # Add this line for debugging
    im = io.imread(image)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=True).type(torch.float32)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():  # Ensure no gradient tracking
        result = model(image)
    result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)
    post_processed_result = post_process_segmentation(result)
    result_3ch = post_processed_result.repeat(3, 1, 1)
    result_3ch_np = result_3ch.permute(1, 2, 0).cpu().detach().numpy()  # Detach gradients before converting to numpy
    result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8
    return result_3ch_np_uint8



def process_folder_yolo(folder_path, destination, yolo_model, isnet_model, names):
    os.makedirs(destination, exist_ok=True)  # Ensure the destination directory exists
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print("Processing file:", file_path)  # Add this line for debugging
            try:
                result = yolo_model.predict(source=file_path, conf=0.20)
                predictions = result[0]
                classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)
                boxes = predictions.boxes.xyxy.cpu().detach().numpy()  # Get bounding boxes

                tshirt_boxes = [box for cls, box in zip(classes, boxes) if cls == 1 and names.get(cls) == "t-shirt"]  # Filter for T-shirts

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                if not save_classified_image_yolo(image, destination, tshirt_boxes, isnet_model):
                    print(f"No T-shirts found for {file_name}, skipping save.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

def post_process_segmentation(segmentation):
    segmentation = segmentation.cpu().numpy().squeeze()
    _, binary_mask = cv2.threshold(segmentation, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological closing to fill small holes
    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents((closed_mask * 255).astype(np.uint8))
    min_size = 100
    cleaned_mask = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            cleaned_mask[labels_im == label] = 1

    return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    source = "demo_datasets/yolo_input"
    destination = "demo_datasets/yolo_output"
    names = {0: "none", 1: "t-shirt", 2: "jacket", 3: "top", 4: "bottom", 5: "shoes", 6: "shirt"}

    yolo_model_path = "saved_models/best.pt"
    isnet_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\only_cropped.pth"

    try:
        yolo_model = YOLO(yolo_model_path, task="detect").to("cuda")
        isnet_model = ISNetDIS()
        if torch.cuda.is_available():
            isnet_model.load_state_dict(torch.load(isnet_model_path))
            isnet_model = isnet_model.cuda()
        else:
            isnet_model.load_state_dict(torch.load(isnet_model_path, map_location="cpu"))
        isnet_model.eval()

        process_images_yolo(source, destination, yolo_model, isnet_model, names)
    except Exception as e:
        print(f"Error loading model or processing images: {e}")
