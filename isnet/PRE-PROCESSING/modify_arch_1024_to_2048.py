import torch
import torch.nn as nn
from isnet_2048 import ISNetDIS

# Load the pretrained model weights (OrderedDict)
pretrained_model_path = '/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/gpu_itr_172000_traLoss_0.0504_traTarLoss_0.0039_valLoss_0.1674_valTarLoss_0.0183_maxF1_0.9956_mae_0.0036_time_0.037569.pth'
pretrained_state_dict = torch.load(pretrained_model_path)


# Initialize the modified model
modified_model = ISNetDIS()

# Remove the existing 'conv_in_0' weights from the pretrained model if they exist
if 'conv_in_0.weight' in pretrained_state_dict:
    pretrained_state_dict.pop('conv_in_0.weight', None)
    pretrained_state_dict.pop('conv_in_0.bias', None)

# Load the weights into the modified model, skipping 'conv_in' layer
pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'conv_in' not in k}

# Load the remaining weights into the modified model
modified_model.load_state_dict(pretrained_state_dict, strict=False)

# Manually initialize 'conv_in_0' if it's not included in the pretrained weights
modified_model.conv_in_0.weight.data.normal_(0, 0.01)
modified_model.conv_in_0.bias.data.zero_()

# Save the modified model checkpoint
modified_model_path = '/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/pretrained_1024_172000_2048.pth'
torch.save(modified_model.state_dict(), modified_model_path)

print("Modified model checkpoint saved successfully!")













# import torch
# from isnet_2048 import ISNetDIS

# model = ISNetDIS()  # Initialize your model (ensure it matches the pretrained one)
# model.load_state_dict(torch.load('/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/Modified_isnet_checkpoint.pth'))  # Load pretrained weights

# # Print the architecture (this will display the model's layers and their details)
# model_architecture = str(model)

# with open('model_architecture_NEW.txt', 'w') as f:
#     f.write(model_architecture)

# print("Model architecture saved to model_architecture.txt")