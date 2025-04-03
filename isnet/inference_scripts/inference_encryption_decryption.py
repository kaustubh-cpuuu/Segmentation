import os
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS  # Make sure this import matches your project's structure
import cv2
from glob import glob
from cryptography.fernet import Fernet

# Function to decrypt the model checkpoint
def decrypt_checkpoint(encrypted_checkpoint_path, encryption_key_path):
    with open(encryption_key_path, 'rb') as key_file:
        key = key_file.read()
    cipher_suite = Fernet(key)
    with open(encrypted_checkpoint_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    temp_checkpoint_path = 'temp_decrypted_checkpoint.pth'
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

if __name__ == "__main__":
    dataset_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset"
    encrypted_model_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\encrypted_checkpoint"
    encryption_key_path = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\saved_models\IS-Net\encryption_key.key"
    result_path = "C:\\Users\\ml2au\\PycharmProjects\\Background_removal\\DIS\\demo_datasets\\your_dataset_result"
    input_size = [1024, 1024]

    # Decrypt the model checkpoint
    decrypted_model_path = decrypt_checkpoint(encrypted_model_path, encryption_key_path)

    net = ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(decrypted_model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(decrypted_model_path, map_location="cpu"))
    net.eval()

    # Cleanup decrypted model checkpoint
    os.remove(decrypted_model_path)

    im_list = glob(dataset_path + "\\*.jpg") + glob(dataset_path + "\\*.jpeg") + glob(dataset_path + "\\*.png") + glob(dataset_path + "\\*.bmp") + glob(dataset_path + "\\*.tiff")

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("Processing image: ", im_path)
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

            im_name = os.path.split(im_path)[-1].split('.')[0]
            io.imsave(os.path.join(result_path, im_name + ".png"), result_3ch_np_uint8)
