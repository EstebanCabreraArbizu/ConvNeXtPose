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
        raise FileNotFoundError(f'No se encontró ningún archivo que comience con "{checkpoint_filename}" en {model_path}')
    for member in members:
        z.extract(member, path = tmp_dir)
    
    # Extraer únicamente el archivo checkpoint
    # z.extract(checkpoint_filename, path=tmp_dir)

# Ahora, los archivos extraídos quedaron con la ruta: tmp_dir/snapshot_68.pth/...
# Para que torch.load pueda interpretarlo, debemos “re-empaquetar” en memoria eliminando el prefijo.
# Nota: Esta operación en memoria preserva el contenido de cada archivo (incluyendo el record "version")
bio = io.BytesIO()
with zipfile.ZipFile(bio, 'w') as newzip:
    for member in members:
        # Lee el contenido del archivo extraído
        file_path = os.path.join(tmp_dir, member)
        with open(file_path, 'rb') as f:
            data = f.read()
        prefix = f"{checkpoint_filename}/"
        if member.startswith(prefix):
            newname = 'archive/'+ member[len(prefix):]  # newname = os.path.relpath(member, checkpoint_filename)
        else: 
            newname = member
        # Remueve el prefijo "snapshot_68.pth/" del nombre interno
        print(f"Empaquetando: {member} como {newname}")  # Verifica los nombres
        newzip.writestr(newname, data)
bio.seek(0)
print(zipfile.ZipFile(bio).namelist())

# Opcional: Guardar el archivo .pth en disco si es necesario
# output_path = f'snapshot_{test_epoch}_repacked.pth'
# with open(output_path, 'wb') as f:
#     f.write(bio.getvalue())

shutil.rmtree(tmp_dir)  # Limpiar el directorio temporal
print(os.getcwd())