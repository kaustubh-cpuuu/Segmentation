import os
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS  # Make sure this import matches your project structure
import cv2
from glob import glob
from cryptography.fernet import Fernet
import tempfile
import shutil

def decrypt_checkpoint(encrypted_checkpoint_path, encryption_key_path):
    with open(encryption_key_path, 'rb') as key_file:
        key = key_file.read()
    cipher_suite = Fernet(key)
    with open(encrypted_checkpoint_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    temp_checkpoint_path = tempfile.mktemp(suffix='.pth')
    with open(temp_checkpoint_path, 'wb') as temp_file:
        temp_file.write(decrypted_data)
    return temp_checkpoint_path

def post_process_segmentation(segmentation):
    segmentation = segmentation.cpu().numpy().squeeze()
    _, binary_mask = cv2.threshold(segmentation, 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels_im = cv2.connectedComponents((closed_mask * 255).astype(np.uint8))
    min_size = 100
    cleaned_mask = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if np.sum(labels_im == label) > min_size:
            cleaned_mask[labels_im == label] = 1
    return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)

def setup_image_folders(base_path, image_name, output_suffix):
    image_folder_path = os.path.join(base_path, image_name)
    input_folder_path = os.path.join(image_folder_path, 'input')
    output_folder_path = os.path.join(image_folder_path, 'output')
    os.makedirs(input_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)
    return input_folder_path, output_folder_path

def process_images(net, im_list, temp_result_path, input_size, output_suffix):
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print(f"Processing image for {output_suffix}: ", im_path)
            im_name = os.path.basename(im_path).split('.')[0]
            input_folder_path, output_folder_path = setup_image_folders(temp_result_path, im_name, output_suffix)

            # Copy input image to the input folder, but only if it's not already there
            if output_suffix == "tshirt":
                shutil.copy(im_path, input_folder_path)

            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=True).type(torch.float32)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if torch.cuda.is_available():
                image = image.cuda()
            result = net(image)
            result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)
            post_processed_result = post_process_segmentation(result)
            result_3ch = post_processed_result.repeat(3, 1, 1)
            result_3ch_np = result_3ch.permute(1, 2, 0).cpu().data.numpy()
            result_3ch_np_uint8 = (result_3ch_np * 255).astype(np.uint8)

            output_filename = f"{im_name}_{output_suffix}.png"
            io.imsave(os.path.join(output_folder_path, output_filename), result_3ch_np_uint8)


if __name__ == "__main__":
    dataset_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset"
    encrypted_model_path1 = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\tshirt_checkpoint"
    encrypted_model_path2 = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\shoes_checkpoint"
    encryption_key_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\encryption_key.key"

    temp_dir = os.path.join(tempfile.gettempdir(), 'results')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_result_path = temp_dir
    print(f"Results will be saved in: {temp_result_path}")

    input_size = [1024, 1024]

    net = ISNetDIS()

    # Load and process images with the first checkpoint, output folder named "tshirt"
    decrypted_model_path1 = decrypt_checkpoint(encrypted_model_path1, encryption_key_path)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(decrypted_model_path1))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(decrypted_model_path1, map_location="cpu"))
    net.eval()
    im_list = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.jpeg")) + glob(
        os.path.join(dataset_path, "*.png")) + glob(os.path.join(dataset_path, "*.bmp")) + glob(
        os.path.join(dataset_path, "*.tiff"))
    process_images(net, im_list, temp_result_path, input_size, "tshirt")
    os.remove(decrypted_model_path1)

    # Load and process images with the second checkpoint, output folder named "shoes"
    decrypted_model_path2 = decrypt_checkpoint(encrypted_model_path2, encryption_key_path)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(decrypted_model_path2))
    else:
        net.load_state_dict(torch.load(decrypted_model_path2, map_location="cpu"))
    net.eval()
    process_images(net, im_list, temp_result_path, input_size, "shoes")
    os.remove(decrypted_model_path2)
