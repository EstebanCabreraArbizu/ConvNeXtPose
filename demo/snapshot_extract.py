import zipfile
import tempfile
import os
import io
import shutil

# Ruta al archivo que contiene varios checkpoints empaquetados
test_epoch = 68
model_path = f'demo/ConvNeXtPose_XS/snapshot_{test_epoch}.pth'


print(model_path.split('\\'))