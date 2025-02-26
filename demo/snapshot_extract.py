import zipfile
import tempfile
import os
import shutil
import torch

# Ruta al archivo que contiene varios checkpoints empaquetados
model_path = 'demo/ConvNeXtPose_XS.tar'
test_epoch = 68
checkpoint_filename = f'snapshot_{test_epoch}.pth'  # sin barra final

with zipfile.ZipFile(model_path) as z:
    # Verifica que el archivo existe (nota: sin la barra final)
    if checkpoint_filename not in z.namelist():
        raise FileNotFoundError(f'Archivo "{checkpoint_filename}" no encontrado en {model_path}')
    # Crear un directorio temporal para extraer el checkpoint completo
    tmp_dir = tempfile.mkdtemp()
    # Extraer únicamente el archivo checkpoint
    z.extract(checkpoint_filename, path=tmp_dir)

# Construir la ruta completa al checkpoint extraído
extracted_checkpoint_path = os.path.join(tmp_dir, checkpoint_filename)
# Ahora torch.load puede leer el archivo directamente, con su estructura interna intacta
ckpt = torch.load(extracted_checkpoint_path, map_location=lambda storage, loc: storage.cuda())
print("Checkpoint cargado correctamente!")

# (Opcional) Guardar o usar el checkpoint según se necesite
model_path_out = 'snapshot_68.pth'
torch.save(ckpt, model_path_out)