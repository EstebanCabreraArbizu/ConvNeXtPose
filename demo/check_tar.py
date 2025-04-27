import os
import zipfile
import sys
import glob

def main():
    print("Diagnóstico de archivos tar/zip para ConvNeXtPose")
    print("="*50)
    
    # Verificar directorio actual
    current_dir = os.getcwd()
    print(f"Directorio actual: {current_dir}")
    
    # Buscar archivos tar en ubicaciones comunes
    search_paths = [
        ".", 
        "./demo", 
        os.path.dirname(__file__), 
        os.path.dirname(os.path.dirname(__file__))
    ]
    
    tar_files = []
    for path in search_paths:
        tar_files.extend(glob.glob(os.path.join(path, "*.tar")))
    
    if not tar_files:
        print("No se encontraron archivos .tar en ninguna ubicación probable")
        print("Asegúrate de haber descargado el modelo desde el enlace proporcionado en el README")
        return
    
    print(f"Archivos .tar encontrados ({len(tar_files)}):")
    for i, tar_file in enumerate(tar_files):
        print(f"{i+1}. {tar_file} - Tamaño: {os.path.getsize(tar_file)/1024/1024:.2f} MB")
    
    # Analizar el contenido de cada archivo
    for i, tar_file in enumerate(tar_files):
        try:
            with zipfile.ZipFile(tar_file, 'r') as z:
                files = z.namelist()
                print(f"\nArchivo {i+1}: {tar_file}")
                print(f"  Total de archivos: {len(files)}")
                
                # Buscar snapshots
                snapshot_files = [f for f in files if f.startswith('snapshot_')]
                print(f"  Snapshots encontrados: {len(snapshot_files)}")
                for snapshot in snapshot_files[:5]:  # Mostrar solo los primeros 5
                    print(f"    - {snapshot}")
                
                if len(snapshot_files) > 5:
                    print(f"    (y {len(snapshot_files)-5} más...)")
        except Exception as e:
            print(f"\nError al analizar {tar_file}: {str(e)}")
    
    print("\nPara usar un archivo específico, modifica MODEL_TAR en real_test_simple.py:")
    print('MODEL_TAR = "ruta/completa/al/archivo.tar"')

if __name__ == "__main__":
    main()
