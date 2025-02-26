import zipfile
import tempfile
import os
import shutil
# snapshot load
model_path = 'demo/ConvNeXtPose_XS.tar'
test_epoch = 68
checkpoint_folder = f'snapshot_{test_epoch}.pth/'
with zipfile.ZipFile(model_path) as z:
    members = [m for m in z.namelist() if m.startswith(checkpoint_folder)]
    if not members:
        raise FileNotFoundError(f'Archivo {checkpoint_folder} no encontrado en {model_path}')
    # Crear un directorio temporal para extraer la carpeta completa
    tmp_dir = tempfile.mkdtemp()
    # Extraer todos los miembros preservando la estructura
    z.extractall(path=tmp_dir)

# Construir la ruta completa al checkpoint extraído
# Suponiendo que el checkpoint extraído es un archivo ZIP interno que torch.load puede interpretar
extracted_checkpoint_dir = os.path.join(tmp_dir, checkpoint_folder)
repacked_checkpoint_path = os.path.join(tmp_dir, 'temp_checkpoint.pth')

# shutil.make_archive crea un ZIP; le quitamos la extensión .zip y luego renombramos el archivo resultante
archive_base = repacked_checkpoint_path[:-4]  # quita ".pth"
shutil.make_archive(archive_base, 'zip', root_dir = extracted_checkpoint_dir, base_dir= ".")
# El archivo generado se llamará archive_base + '.zip'. Lo renombramos a .pth
os.rename(archive_base + '.zip', repacked_checkpoint_path)

print(zipfile.ZipFile(repacked_checkpoint_path).namelist())