
from cryptography.fernet import Fernet
import os

# Define your paths
checkpoint_path = 'saved_models/IS-Net/IS-Netisnet.pth'
directory_path = '../saved_models/IS-Net'
encrypted_file_path = f'{directory_path}/shoes_checkpoint'
key_file_path = f'{directory_path}/encryption_key.key'

# Step 1: Check if the encryption key already exists
if os.path.exists(key_file_path):
    with open(key_file_path, 'rb') as key_file:
        key = key_file.read()
else:
    # Generate a new key if it doesn't exist
    key = Fernet.generate_key()
    with open(key_file_path, 'wb') as key_file:
        key_file.write(key)

# Initialize the cipher suite with the existing or new key
cipher_suite = Fernet(key)

# Step 2: Encrypt the checkpoint
with open(checkpoint_path, 'rb') as file:
    checkpoint_data = file.read()
encrypted_data = cipher_suite.encrypt(checkpoint_data)

# Step 3: Save the encrypted data
with open(encrypted_file_path, 'wb') as file:
    file.write(encrypted_data)

print(f'Encrypted checkpoint saved to {encrypted_file_path}')
print(f'Encryption key used is located at {key_file_path}')
