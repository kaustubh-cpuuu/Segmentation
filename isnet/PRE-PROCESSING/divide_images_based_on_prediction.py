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


def crop_and_save_image(image, box, expansion_pixels, destination_path):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 = max(0, x1 - expansion_pixels)
    y1 = max(0, y1 - expansion_pixels)
    x2 = min(image.shape[1], x2 + expansion_pixels)
    y2 = min(image.shape[0], y2 + expansion_pixels)

    cropped_image = image[y1:y2, x1:x2]
    if cropped_image.size == 0:
        print("Empty cropped image, skipping.")
        return False

    cv2.imwrite(destination_path, cropped_image)
    return True


def save_classified_image_yolo(image, destination, boxes, isnet_model, file_name):
    saved = False
    for i, box in enumerate(boxes):  # Assuming multiple T-shirts could be present
        cropped_image_path = f"tshirt_{i}.png"
        cropped_image = crop_and_save_image(image, box, 20, cropped_image_path)
        if cropped_image:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            expanded_x1 = max(0, x1 - 20)
            expanded_y1 = max(0, y1 - 20)
            expanded_x2 = min(image.shape[1], x2 + 20)
            expanded_y2 = min(image.shape[0], y2 + 20)

            expanded_cropped_image = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            expanded_height = expanded_y2 - expanded_y1
            expanded_width = expanded_x2 - expanded_x1

            if expanded_height > 5500 or expanded_width > 2800:
                if expanded_height > 5500 and expanded_width > 2800:
                    part_height = expanded_height // 3

                    first_part = expanded_cropped_image[:part_height, :]
                    second_part = expanded_cropped_image[part_height:2 * part_height, :]
                    third_part = expanded_cropped_image[2 * part_height:, :]

                    first_part_width = first_part.shape[1]
                    second_part_width = second_part.shape[1]
                    third_part_width = third_part.shape[1]

                    first_part_first_half = first_part[:, :first_part_width // 2]
                    first_part_second_half = first_part[:, first_part_width // 2:]

                    second_part_first_half = second_part[:, :second_part_width // 2]
                    second_part_second_half = second_part[:, second_part_width // 2:]

                    third_part_first_half = third_part[:, :third_part_width // 2]
                    third_part_second_half = third_part[:, third_part_width // 2:]

                    first_part_first_half_expanded_image_path = f"first_part_first_half_tshirt_{i}.png"
                    first_part_second_half_expanded_image_path = f"first_part_second_half_tshirt_{i}.png"
                    second_part_first_half_expanded_image_path = f"second_part_first_half_tshirt_{i}.png"
                    second_part_second_half_expanded_image_path = f"second_part_second_half_tshirt_{i}.png"
                    third_part_first_half_expanded_image_path = f"third_part_first_half_tshirt_{i}.png"
                    third_part_second_half_expanded_image_path = f"third_part_second_half_tshirt_{i}.png"

                    cv2.imwrite(first_part_first_half_expanded_image_path, first_part_first_half)
                    cv2.imwrite(first_part_second_half_expanded_image_path, first_part_second_half)
                    cv2.imwrite(second_part_first_half_expanded_image_path, second_part_first_half)
                    cv2.imwrite(second_part_second_half_expanded_image_path, second_part_second_half)
                    cv2.imwrite(third_part_first_half_expanded_image_path, third_part_first_half)
                    cv2.imwrite(third_part_second_half_expanded_image_path, third_part_second_half)

                    first_part_first_half_isnet_result = process_image_isnet(first_part_first_half_expanded_image_path,
                                                                             isnet_model)
                    first_part_second_half_isnet_result = process_image_isnet(
                        first_part_second_half_expanded_image_path, isnet_model)
                    second_part_first_half_isnet_result = process_image_isnet(
                        second_part_first_half_expanded_image_path, isnet_model)
                    second_part_second_half_isnet_result = process_image_isnet(
                        second_part_second_half_expanded_image_path, isnet_model)
                    third_part_first_half_isnet_result = process_image_isnet(third_part_first_half_expanded_image_path,
                                                                             isnet_model)
                    third_part_second_half_isnet_result = process_image_isnet(
                        third_part_second_half_expanded_image_path, isnet_model)

                    first_part_isnet_result = np.concatenate(
                        (first_part_first_half_isnet_result, first_part_second_half_isnet_result), axis=1)
                    second_part_isnet_result = np.concatenate(
                        (second_part_first_half_isnet_result, second_part_second_half_isnet_result), axis=1)
                    third_part_isnet_result = np.concatenate(
                        (third_part_first_half_isnet_result, third_part_second_half_isnet_result), axis=1)

                    combined_isnet_result = np.concatenate(
                        (first_part_isnet_result, second_part_isnet_result, third_part_isnet_result), axis=0)

                    cv2.imwrite(os.path.join(destination, f"first_first_part_tshirt_{i}_{file_name}"),
                                first_part_first_half)
                    cv2.imwrite(os.path.join(destination, f"first_second_part_tshirt_{i}_{file_name}"),
                                first_part_second_half)
                    cv2.imwrite(os.path.join(destination, f"second_first_part_tshirt_{i}_{file_name}"),
                                second_part_first_half)
                    cv2.imwrite(os.path.join(destination, f"second_second_part_tshirt_{i}_{file_name}"),
                                second_part_second_half)
                    cv2.imwrite(os.path.join(destination, f"third_first_part_tshirt_{i}_{file_name}"),
                                third_part_first_half)
                    cv2.imwrite(os.path.join(destination, f"third_second_part_tshirt_{i}_{file_name}"),
                                third_part_second_half)

                elif expanded_height > 5500 and expanded_width < 2800:
                    part_height = expanded_height // 3

                    first_part = expanded_cropped_image[:part_height, :]
                    second_part = expanded_cropped_image[part_height:2 * part_height, :]
                    third_part = expanded_cropped_image[2 * part_height:, :]

                    first_part_expanded_image_path = f"first_part_tshirt_{i}.png"
                    second_part_expanded_image_path = f"second_part_tshirt_{i}.png"
                    third_part_expanded_image_path = f"third_part_tshirt_{i}.png"

                    cv2.imwrite(first_part_expanded_image_path, first_part)
                    cv2.imwrite(second_part_expanded_image_path, second_part)
                    cv2.imwrite(third_part_expanded_image_path, third_part)

                    first_part_isnet_result = process_image_isnet(first_part_expanded_image_path, isnet_model)
                    second_part_isnet_result = process_image_isnet(second_part_expanded_image_path, isnet_model)
                    third_part_isnet_result = process_image_isnet(third_part_expanded_image_path, isnet_model)

                    combined_isnet_result = np.concatenate((first_part_isnet_result,
                                                            second_part_isnet_result,
                                                            third_part_isnet_result), axis=0)

                    # Save cropped parts
                    cv2.imwrite(os.path.join(destination, f"first_part_tshirt_{i}_{file_name}"), first_part)
                    cv2.imwrite(os.path.join(destination, f"second_part_tshirt_{i}_{file_name}"), second_part)
                    cv2.imwrite(os.path.join(destination, f"third_part_tshirt_{i}_{file_name}"), third_part)

                elif expanded_height < 5500 and  expanded_width < 2800:
                    part_height = expanded_height // 2

                    first_part = expanded_cropped_image[:part_height, :]
                    second_part = expanded_cropped_image[part_height:, :]

                    first_part_width = first_part.shape[1]
                    second_part_width = second_part.shape[1]

                    first_part_first_half = first_part[:, :first_part_width //2]
                    first_part_second_half = first_part[: ,first_part_width // 2:]

                    second_part_first_half = second_part[:, :second_part_width // 2]
                    second_part_second_half = second_part[:, second_part_width // 2:]

                    first_part_first_half_expanded_image_path = f"first_part_first_half_tshirt_{i}.png"
                    first_part_second_half_expanded_image_path = f"first_part_second_half_tshirt_{i}.png"
                    second_part_first_half_expanded_image_path = f"second_part_first_half_tshirt_{i}.png"
                    second_part_second_half_expanded_image_path = f"second_part_second_half_tshirt_{i}.png"

                    cv2.imwrite(first_part_first_half_expanded_image_path, first_part_first_half)
                    cv2.imwrite(first_part_second_half_expanded_image_path, first_part_second_half)
                    cv2.imwrite(second_part_first_half_expanded_image_path, second_part_first_half)
                    cv2.imwrite(second_part_second_half_expanded_image_path, second_part_second_half)

                    first_part_first_half_isnet_result = process_image_isnet(first_part_first_half_expanded_image_path,isnet_model)
                    first_part_second_half_isnet_result = process_image_isnet(first_part_second_half_expanded_image_path, isnet_model)
                    second_part_first_half_isnet_result = process_image_isnet(second_part_first_half_expanded_image_path, isnet_model)
                    second_part_second_half_isnet_result = process_image_isnet(second_part_second_half_expanded_image_path, isnet_model)

                    first_part_isnet_result = np.concatenate((first_part_first_half_isnet_result, first_part_second_half_isnet_result), axis=1)
                    second_part_isnet_result = np.concatenate((second_part_first_half_isnet_result, second_part_second_half_isnet_result), axis=1)

                    combined_isnet_result = np.concatenate(
                        (first_part_isnet_result, second_part_isnet_result), axis=0)

                    cv2.imwrite(os.path.join(destination, f"first_first_part_tshirt_{i}_{file_name}"),
                                first_part_first_half)
                    cv2.imwrite(os.path.join(destination, f"first_second_part_tshirt_{i}_{file_name}"),
                                first_part_second_half)
                    cv2.imwrite(os.path.join(destination, f"second_first_part_tshirt_{i}_{file_name}"),
                                second_part_first_half)
                    cv2.imwrite(os.path.join(destination, f"second_second_part_tshirt_{i}_{file_name}"),
                                second_part_second_half)

                elif expanded_width > 3000 and expanded_height > 2800:
                    part_width = expanded_width // 2

                    first_part = expanded_cropped_image[:, :part_width]
                    second_part = expanded_cropped_image[:, part_width:]

                    first_part_expanded_image_path = f"first_part_tshirt_{i}.png"
                    second_part_expanded_image_path = f"second_part_tshirt_{i}.png"

                    cv2.imwrite(first_part_expanded_image_path, first_part)
                    cv2.imwrite(second_part_expanded_image_path, second_part)

                    first_part_isnet_result = process_image_isnet(first_part_expanded_image_path, isnet_model)
                    second_part_isnet_result = process_image_isnet(second_part_expanded_image_path, isnet_model)

                    combined_isnet_result = np.concatenate((first_part_isnet_result, second_part_isnet_result), axis=1)

                    cv2.imwrite(os.path.join(destination, f"first_part_tshirt_{i}_{file_name}"), first_part)
                    cv2.imwrite(os.path.join(destination, f"second_part_tshirt_{i}_{file_name}"), second_part)

            elif expanded_width < 2500 and expanded_height < 2500:

                first_part = expanded_cropped_image

                first_part_expanded_image_path = f"first_part_tshirt_{i}.png"
                cv2.imwrite(first_part_expanded_image_path, first_part)
                first_part_isnet_result = process_image_isnet(first_part_expanded_image_path, isnet_model)
                combined_isnet_result = first_part_isnet_result

                cv2.imwrite(os.path.join(destination, f"first_part_tshirt_{i}_{file_name}"), first_part)

            else:
                part_height = expanded_height // 2

                first_part = expanded_cropped_image[:part_height, :]
                second_part = expanded_cropped_image[part_height:, :]

                # Process each part separately
                first_part_expanded_image_path = f"first_part_tshirt_{i}.png"
                second_part_expanded_image_path = f"second_part_tshirt_{i}.png"

                cv2.imwrite(first_part_expanded_image_path, first_part)
                cv2.imwrite(second_part_expanded_image_path, second_part)

                first_part_isnet_result = process_image_isnet(first_part_expanded_image_path, isnet_model)
                second_part_isnet_result = process_image_isnet(second_part_expanded_image_path, isnet_model)

                combined_isnet_result = np.concatenate((first_part_isnet_result, second_part_isnet_result), axis=0)

                cv2.imwrite(os.path.join(destination, f"first_part_tshirt_{i}_{file_name}"), first_part)
                cv2.imwrite(os.path.join(destination, f"second_part_tshirt_{i}_{file_name}"), second_part)

            black_image = np.zeros_like(image)
            black_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = combined_isnet_result

            result_path = os.path.join(destination, file_name)
            cv2.imwrite(result_path, black_image)
            cv2.imwrite(os.path.join(destination, f"    ORIGINAL_tshirt_{i}_{file_name}"),
                        cv2.imread(cropped_image_path))
            saved = True

    return saved

def process_image_isnet(image_path, model):
    input_size = [1024, 1024]
    im = io.imread(image_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=True).type(
        torch.float32)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        result = model(image)
    result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)

    post_processed_result = post_process_segmentation(result)
    result_3ch = post_processed_result.repeat(3, 1, 1)
    result_3ch_np = result_3ch.permute(1, 2, 0).cpu().detach().numpy()
    return (result_3ch_np * 255).astype(np.uint8)


def post_process_segmentation(segmentation):
    segmentation = segmentation.cpu().numpy().squeeze()
    _, binary_mask = cv2.threshold(segmentation, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological closing to fill small holes
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents((closed_mask * 255).astype(np.uint8))
    min_size = 100
    cleaned_mask = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            cleaned_mask[labels_im == label] = 1
    return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)


def process_folder_yolo(folder_path, destination, yolo_model, isnet_model, names):
    os.makedirs(destination, exist_ok=True)
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                result = yolo_model.predict(source=file_path, conf=0.50, iou=0.50)
                predictions = result[0]
                classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)
                boxes = predictions.boxes.xyxy.cpu().detach().numpy()

                tshirt_boxes = [box for cls, box in zip(classes, boxes) if cls == 0 and names.get(cls) == "person"]

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                file_name_only = os.path.splitext(file_name)[0] + ".png"
                if not save_classified_image_yolo(image, destination, tshirt_boxes, isnet_model, file_name_only):
                    print(f"No T-shirts found for {file_name}, skipping save.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


def process_images_yolo(source, destination, yolo_model, isnet_model, names):
    if not os.path.exists(source):
        print(f"Source directory {source} does not exist.")
        return

    if not os.path.exists(destination):
        os.makedirs(destination)

    for folder_name in os.listdir(source):
        folder_path = os.path.join(source, folder_name)
        if os.path.isdir(folder_path):
            process_folder_yolo(folder_path, destination, yolo_model, isnet_model, names)
        else:
            process_folder_yolo(source, destination, yolo_model, isnet_model, names)
            break


if __name__ == "__main__":
    source = "demo_datasets/yolo_input"
    destination = "demo_datasets/yolo_output"
    names = {0: "person", 1: "t-shirt", 2: "shoes", 3: "bottom", 4: "inner", 5: "onepiece"}

    yolo_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\product_yolo.pt"
    isnet_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\product.pth"

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

