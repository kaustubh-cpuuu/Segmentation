import torch
import torch.onnx
from isnet_DenseNet_121 import ISNetDIS


model = ISNetDIS()
checkpoint_path = 'saved_models/1300_new_impro_hbx_full_crop.pth'
checkpoint = torch.load(checkpoint_path)
# print("Checkpoint keys:", checkpoint.keys())

try:
    model.load_state_dict(checkpoint['model'])
except KeyError:
    model.load_state_dict(checkpoint)
model.eval()
dummy_input = torch.randn(1, 3, 1024, 1024, requires_grad=True)
onnx_file_path = 'saved_models/1300_new_impro_hbx_full_crop.onnx'


torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f'Model has been converted to ONNX and saved to {onnx_file_path}')
