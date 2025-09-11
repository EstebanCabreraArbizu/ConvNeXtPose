# Tu código original con opset_version ajustado
from config import cfg
from model import get_pose_net
import torch
import os
from torch.nn.parallel import DataParallel

# Construir modelo
model_type = 'XS'  # Cambia a 'S', 'M', o 'L' según el modelo que desees convertir
model_path = f'../demo/ConvNeXtPose_{model_type}.tar'  # Ruta al modelo entrenado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(model_path, map_location=device)
model = get_pose_net(cfg, is_train=False, joint_num=18)

if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = DataParallel(model)
    model = model.to(device)
    # Cargar state dict con manejo de DataParallel
    try:
        model.load_state_dict(ckpt['network'], strict=False)
    except:
        # Si falla, intentar sin DataParallel wrapper
        state_dict = {}
        for key, value in ckpt['network'].items():
            if key.startswith('module.'):
                state_dict[key[7:]] = value  # Remove 'module.' prefix
            else:
                state_dict[f'module.{key}'] = value  # Add 'module.' prefix
        model.load_state_dict(state_dict, strict=False)
else:
    model = model.to(device)
    # Para modelo sin DataParallel
    state_dict = ckpt['network']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix si existe
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    f'convnextpose_{model_type}.onnx',
    export_params=True,
    opset_version=11,  # Mantener 11 para onnx-tf 1.10.0
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# Conversión ONNX → TFLite
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load(f'convnextpose_{model_type}.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('tmp_tf_model')

converter = tf.lite.TFLiteConverter.from_saved_model('tmp_tf_model')
tflite_model = converter.convert()
with open(f'convnextpose_{model_type}.tflite', 'wb') as f:
    f.write(tflite_model)