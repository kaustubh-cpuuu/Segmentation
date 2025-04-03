import torch
from collections import OrderedDict


def adjust_checkpoint_keys(checkpoint_path, adjusted_checkpoint_path):
    # Load the original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check if the checkpoint is a state_dict or contains a state_dict under a key (commonly 'state_dict')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Create a new state_dict without the 'module.' prefix
    adjusted_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        adjusted_state_dict[name] = v

    # If the original checkpoint had a 'state_dict' key, replace it with the adjusted one
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = adjusted_state_dict
    else:
        checkpoint = adjusted_state_dict

    # Save the adjusted checkpoint
    torch.save(checkpoint, adjusted_checkpoint_path)
    print(f"Adjusted checkpoint saved to {adjusted_checkpoint_path}")


# Specify the original and the new checkpoint paths
original_checkpoint_path = "prev_checkpoints/cloth_segm_unet_surgery.pth"
adjusted_checkpoint_path = 'prev_checkpoints/cloth_segm.pth'

# Adjust the checkpoint keys
adjust_checkpoint_keys(original_checkpoint_path, adjusted_checkpoint_path)
