import torch
import torch.onnx
from isnet_2048 import ISNetDIS 

model = ISNetDIS()

checkpoint_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/_gpu_itr_70000_traLoss_0.3919_traTarLoss_0.0481_valLoss_0.4004_valTarLoss_0.0531_maxF1_0.5528_mae_0.0249_time_0.021894-70000.pth"
checkpoint = torch.load(checkpoint_path)
# print("Checkpoint keys:", checkpoint.keys())



try:
    model.load_state_dict(checkpoint["model"])
except KeyError:
    model.load_state_dict(checkpoint)
model.eval()
dummy_input = torch.randn(1, 3, 2048, 2048, requires_grad=True)
onnx_file_path = "saved_models/new_dust_1600im_70000_itr.onnx"



torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,         
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)



print(f"Model has been converted to ONNX and saved to {onnx_file_path}")





#Mixed precision
# from onnxconverter_common import float16
# import onnx

# onnx_model_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/2048_final_72.onnx"
# onnx_model = onnx.load(onnx_model_path)

# onnx_fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)


# onnx_fp16_model_path = "/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/model_fp16.onnx"
# onnx.save(onnx_fp16_model, onnx_fp16_model_path)

# print(f"Selective FP16 ONNX model saved at: {onnx_fp16_model_path}")



