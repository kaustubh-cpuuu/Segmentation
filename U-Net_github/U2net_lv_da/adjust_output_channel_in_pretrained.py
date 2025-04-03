import torch
import torch.nn as nn
# Assuming `U2NET` is the class name of your model. You need to replace it with the correct class name and import statement
from networks.u2net import U2NET

# Load the pretrained model checkpoint
checkpoint_path = 'trained_checkpoint/cloth_segm.pth'
model = U2NET()  # Initialize your model
checkpoint = torch.load(checkpoint_path)
# print(checkpoint.keys())


# Modify the output layer
# This part is highly dependent on your U-2-Net model's architecture.
# Assuming the output layer is the last layer of the model and named 'outconv' or similar
# For U-2-Net, you might need to adjust the output of each side output layer if it's structured for deep supervision
num_output_channels = 6  # New number of classes
model.outconv = nn.Conv2d(in_channels=model.outconv.in_channels,
                          out_channels=num_output_channels,
                          kernel_size=model.outconv.kernel_size,
                          stride=model.outconv.stride,
                          padding=model.outconv.padding)

# Print the modified model architecture to verify the change
print(model)

# Save the modified model checkpoint
new_checkpoint_path = 'trained_checkpoint/modified_cloth_segm.pth'
torch.save(model.state_dict(), new_checkpoint_path)

new_checkpoint_path
