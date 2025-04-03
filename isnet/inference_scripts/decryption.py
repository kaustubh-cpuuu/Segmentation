
import torch
from cryptography.fernet import Fernet
from isnet import  ISNetDIS
import os

# Define the paths to your encrypted checkpoint and encryption key
encrypted_checkpoint_path = 'saved_models/IS-Net/shoes_checkpoint'
encryption_key_path = 'saved_models/IS-Net/encryption_key.key'

# Load the encryption key
with open(encryption_key_path, 'rb') as key_file:
    key = key_file.read()

cipher_suite = Fernet(key)

# Decrypt the checkpoint
with open(encrypted_checkpoint_path, 'rb') as encrypted_file:
    encrypted_data = encrypted_file.read()

decrypted_data = cipher_suite.decrypt(encrypted_data)

# Assuming the decrypted data is a PyTorch model checkpoint,
# we need to save it temporarily to load it into the model.
# You can choose to keep it in memory if preferred, but here we demonstrate writing and reading from a file.
temp_checkpoint_path = 'temp_decrypted_checkpoint.pth'
with open(temp_checkpoint_path, 'wb') as temp_file:
    temp_file.write(decrypted_data)

# Load the checkpoint into your model
# Ensure your model architecture is defined and instantiated
model = ISNetDIS()  # Your model definition here
model.load_state_dict(torch.load(temp_checkpoint_path, map_location=torch.device('cpu')))
model.eval()

# Cleanup: remove the temporary checkpoint file if you don't need it anymore

os.remove(temp_checkpoint_path)

# Now, your model is ready to be used with the decrypted checkpoint.
