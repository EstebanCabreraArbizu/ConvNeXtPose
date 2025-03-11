import zipfile
import tempfile
import os
import io
import shutil

# Ruta al archivo que contiene varios checkpoints empaquetados
model_path = 'demo/ConvNeXtPose_XS.tar'
test_epoch = 68
checkpoint_filename = f'snapshot_{test_epoch}.pth'  # sin barra final
# Crear un directorio temporal para extraer el checkpoint completo
tmp_dir = tempfile.mkdtemp()

with zipfile.ZipFile(model_path) as z:
    members = [m for m in z.namelist() if m.startswith(checkpoint_filename)]
    if not members:
        raise FileNotFoundError(f'Archivo que comienza con "{checkpoint_filename}" no encontrado en {model_path}')
    for member in members:
        z.extract(member, path=tmp_dir)

extracted_checkpoint_path = os.path.join(tmp_dir, checkpoint_filename)
print(f'Archivo extraído: {extracted_checkpoint_path}')
print(extracted_checkpoint_path.split('/'))