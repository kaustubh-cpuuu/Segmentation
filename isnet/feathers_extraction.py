


# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# from PIL import Image
# import cv2
# import numpy as np
# from isnet import ISNetDIS

# # Define the hook function to capture the output of the layers
# def hook_fn(module, input, output):
#     features.append(output.detach())  # Detach from computation graph to save memory

# # Initialize your custom model
# encoder = ISNetDIS()

# # List to store the feature maps (reset for every inference)
# features = []

# # Register hooks for feature extraction
# layers_to_hook = [
#     encoder.stage1.rebnconv1,
#     encoder.stage2.rebnconv1,
#     encoder.stage3.rebnconv1,
#     encoder.stage4.rebnconv1,
#     encoder.stage5.rebnconv1,
#     encoder.stage6.rebnconv1
# ]

# for layer in layers_to_hook:
#     layer.register_forward_hook(hook_fn)

# # Load and preprocess the input image
# img_path = '/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/input/2nx7ZQsV6B16dXoHVqInHCg5OC8abwYcjIIyzkFhceENvXpSDSkPLL4t35vCOqd820240928141950.jpg'  # Replace with your image path
# img = Image.open(img_path).convert("RGB")  # Ensure it's RGB


# transform = transforms.Compose([
#     transforms.Resize((1024, 1024)),  # Adjust size based on model input
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])  # Model-specific normalization
# ])

# input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# # Move tensor to the same device as the model (if using GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder.to(device)
# input_tensor = input_tensor.to(device)

# # Reset the features list before running inference
# features.clear()

# # Run the model
# output = encoder(input_tensor)

# # Set up subplot grid
# num_layers = len(features)
# num_features_per_layer = 6  # Number of feature maps to display per layer

# fig, axes = plt.subplots(num_layers, num_features_per_layer, figsize=(20, num_layers * 2))

# # Ensure axes is always a 2D array
# if num_layers == 1:
#     axes = [axes]  # Wrap in a list if there's only one layer

# # Loop through each feature map and plot it
# for layer_idx, feature in enumerate(features):
#     feature = feature[0]  # Select first batch sample
#     num_features = feature.shape[0]  # Number of feature channels

#     # Select up to `num_features_per_layer` feature maps
#     num_display = min(num_features, num_features_per_layer)

#     for i in range(num_display):
#         axes[layer_idx, i].imshow(feature[i].cpu().numpy(), cmap='viridis')
#         axes[layer_idx, i].axis("off")

# # Remove empty subplots
# for layer_idx in range(num_layers):
#     for i in range(num_features_per_layer):
#         if i >= features[layer_idx].shape[0]:  # If there's no feature map, hide the subplot
#             axes[layer_idx, i].axis("off")

# # Save the figure
# output_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/feature_maps_grid.png"
# plt.tight_layout()
# plt.savefig(output_path, bbox_inches="tight")
# plt.show()


#Visualize each layer

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from isnet import ISNetDIS
# -----------------------------
# 1. Load and Preprocess Image
# -----------------------------
def load_and_preprocess(image_path):
    """Loads an image, resizes it, converts to tensor, and normalizes."""
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])  # Normalize
    ])
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

# ------------------------------------
# 2. Define Feature Extraction Hook
# ------------------------------------
class FeatureExtractor:
    """Extracts features from a specific layer."""
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.feature_map = None
        self.hook = None
        self.register_hook()

    def register_hook(self):
        """Registers a forward hook on the selected layer."""
        def hook_fn(module, input, output):
            self.feature_map = output.detach().cpu().numpy()  # Save features
        
        layer = dict(self.model.named_modules())[self.layer_name]  # Get the layer
        self.hook = layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        """Removes the registered hook."""
        if self.hook:
            self.hook.remove()

# ------------------------
# 3. Load Trained Model
# ------------------------


# Initialize Model
model = ISNetDIS()  # Replace with actual model
model.load_state_dict(torch.load("/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/IS-Net/isnet-general-use.pth"))  # Load trained weights
model.eval()  # Set to evaluation mode

# ---------------------------------
# 4. Extract and Save Features
# ---------------------------------
layer_name = "stage6"  
extractor = FeatureExtractor(model, layer_name)

# Directory to save extracted features
save_dir = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/features"
os.makedirs(save_dir, exist_ok=True)

# Process images and extract features
input_images = ["/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/input/5LltorwL6b3RhrHIwbgkvygYbmTXfJaAseLZR6dMn9ZRlFBr219rq6PyfcPVsZpU20240928141950.jpg"]  # Add paths to your images

for img_path in input_images:
    img_name = os.path.basename(img_path).split('.')[0]
    
    # Load and preprocess image
    image = load_and_preprocess(img_path)

    # Forward pass
    with torch.no_grad():
        model(image)

    # Check if feature extraction worked
    if extractor.feature_map is None:
        print(f"❌ Feature extraction failed for {img_name}")
        continue  # Skip saving if no features

    print(f"✅ Extracted features for {img_name}, Shape: {extractor.feature_map.shape}")

    # Save features
    feature_map = extractor.feature_map
    np.save(os.path.join(save_dir, f"{img_name}.npy"), feature_map)

extractor.remove_hook()

# ---------------------------
# 5. Visualize Feature Maps
# ---------------------------
def visualize_feature_map(features, num_channels=6):
    """Visualizes a few feature maps."""
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
    for i in range(num_channels):
        axes[i].imshow(features[0, i, :, :], cmap='viridis')  # Display feature map
        axes[i].axis('off')
    plt.show()

# Load and visualize extracted features
features = np.load("/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/features/5LltorwL6b3RhrHIwbgkvygYbmTXfJaAseLZR6dMn9ZRlFBr219rq6PyfcPVsZpU20240928141950.npy")
print("Feature Shape:", features.shape)  # Shape will be (1, C, H, W)
visualize_feature_map(features)

