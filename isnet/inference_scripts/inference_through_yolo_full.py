import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS
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

def save_classified_image_yolo(image, destination, boxes, isnet_model, file_name):
    saved = False
    for i, box in enumerate(boxes):  # Assuming multiple T-shirts could be present
        expansion_pixels = 20

        cropped_image_path = f"tshirt_{i}.png"
        cropped_image = crop_and_save_image(image, box, expansion_pixels, cropped_image_path)
        if cropped_image:
            x1, y1, x2, y2 = [int(coord) for coord in box]

            expanded_x1 = max(0, x1 - expansion_pixels)
            expanded_y1 = max(0, y1 - expansion_pixels)
            expanded_x2 = min(image.shape[1], x2 + expansion_pixels)
            expanded_y2 = min(image.shape[0], y2 + expansion_pixels)

            # Crop the expanded region
            expanded_cropped_image = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

            # Calculate dimensions for halves
            expanded_height = expanded_y2 - expanded_y1
            midpoint = expanded_height // 2

            # Split the expanded cropped image into two halves
            first_half_expanded_image = expanded_cropped_image[:midpoint, :]
            second_half_expanded_image = expanded_cropped_image[midpoint:, :]

            # Ensure both halves have the same width
            min_width = min(first_half_expanded_image.shape[1], second_half_expanded_image.shape[1])
            first_half_expanded_image = first_half_expanded_image[:, :min_width]
            second_half_expanded_image = second_half_expanded_image[:, :min_width]

            # Paths for saving the halves
            first_half_expanded_image_path = f"first_half_tshirt_{i}.png"
            second_half_expanded_image_path = f"second_half_tshirt_{i}.png"

            # Save the first and second halves
            cv2.imwrite(first_half_expanded_image_path, first_half_expanded_image)
            cv2.imwrite(second_half_expanded_image_path, second_half_expanded_image)

            # Process each half expanded image through the ISNet model
            first_half_isnet_result = process_image_isnet(first_half_expanded_image_path, isnet_model)
            second_half_isnet_result = process_image_isnet(second_half_expanded_image_path, isnet_model)

            # Concatenate the results along axis 0 (vertical concatenation)
            combined_isnet_result = np.concatenate((first_half_isnet_result, second_half_isnet_result), axis=0)

            # Create a black image with the original dimensions of the input image
            black_image = np.zeros_like(image)

            # Stitch the combined ISNet result onto the black image
            black_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = combined_isnet_result

            result_path = os.path.join(destination, file_name)
            cv2.imwrite(result_path, black_image)

            saved = True

            original_cropped_path = os.path.join(destination, f"original_cropped_tshirt_{i}_{file_name}")
            cv2.imwrite(original_cropped_path, cv2.imread(cropped_image_path))

    return saved

def process_image_isnet(image, model):
    input_size = [1024,1024]
    # input_size = [512 , 512]
    # print("Processing image:", image)  # Add this line for debugging
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

    # """With post processing"""
    # post_processed_result = post_process_segmentation(result)
    # result_3ch = post_processed_result.repeat(3, 1, 1)
    # result_3ch_np = result_3ch.permute(1, 2, 0).cpu().detach().numpy()  # Detach gradients before converting to numpy
    # result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8
    # return result_3ch_np_uint8

    # """Without preprocessing"""
    result_3ch = result.repeat(3, 1, 1)
    result_3ch_np = result_3ch.permute(1, 2, 0).cpu().detach().numpy()  # Detach gradients before converting to numpy
    result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)  # Convert to uint8

    # result_3ch_np_uint8[(result_3ch_np_   uint8 < [34, 34, 34]).all(axis=2)] = [0, 0, 0]
    # result_3ch_np_uint8[(result_3ch_np_uint8 > [220, 220, 220]).all(axis=2)] = [255, 255, 255]


    # mask = (result_3ch_np_uint8 < [50, 50, 50]).all(axis=2) & ~(
    #             (result_3ch_np_uint8 >= [0, 0, 0]).all(axis=2) & (result_3ch_np_uint8 <= [9, 9, 9]).all(axis=2))
    # decrease_mask = np.stack([mask, mask, mask], axis=-1)
    # result_3ch_np_uint8[decrease_mask] -= 10
    # result_3ch_np_uint8[result_3ch_np_uint8 < 0] = 0
    #
    # mask = (result_3ch_np_uint8 > [220, 220, 220]).all(axis=2) & ~(
    #         (result_3ch_np_uint8 >= [250, 250, 250]).all(axis=2) & (result_3ch_np_uint8 <= [255, 255, 255]).all(axis=2))
    # decrease_mask = np.stack([mask, mask, mask], axis=-1)
    # result_3ch_np_uint8[decrease_mask] += 5
    # result_3ch_np_uint8[result_3ch_np_uint8 < 0] = 255
    #
    # midrange_mask = (result_3ch_np_uint8 >= [200, 200, 200]).all(axis=2) & (result_3ch_np_uint8 <= [245, 245, 245]).all(axis=2)
    # increase_midrange_mask = np.stack([midrange_mask, midrange_mask, midrange_mask], axis=-1)
    # result_3ch_np_uint8[increase_midrange_mask] += 5
    # result_3ch_np_uint8[result_3ch_np_uint8 > 255] = 255

    return result_3ch_np_uint8

def process_folder_yolo(folder_path, destination, yolo_model, isnet_model, names):
    os.makedirs(destination, exist_ok=True)  # Ensure the destination directory exists
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # print("Processing file:", file_path)  # Add this line for debugging
            try:
                result = yolo_model.predict(source=file_path, conf=0.50 , iou=0.50 )
                predictions = result[0]
                classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)
                boxes = predictions.boxes.xyxy.cpu().detach().numpy()  # Get bounding boxes

                """Define_your_detection Class"""
                tshirt_boxes = [box for cls, box in zip(classes, boxes) if cls == 0 and names.get(cls) == "person"]  # Filter for T-shirts

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                file_name_only = os.path.splitext(file_name)[0] + ".png"
                if not save_classified_image_yolo(image, destination, tshirt_boxes, isnet_model, file_name_only):
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
    # names = {0: "none", 1: "t-shirt", 2: "jacket", 3: "top", 4: "bottom", 5: "shoes", 6: "shirt"}
    # names = {0: "t-shirt", 1: "bottom", 2: "shoes" , 3:"full"}
    names = {0: "person", 1: "t-shirt", 2: "shoes", 3: "bottom" ,4:"inner" , 5:"onepiece"}

    # check resolution
    yolo_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\yolov8m.pt"
    isnet_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\background_removal_onmodel.pth"


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

