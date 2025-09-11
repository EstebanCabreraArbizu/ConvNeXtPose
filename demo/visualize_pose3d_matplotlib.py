import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Conexiones de esqueleto para 18 joints (ejemplo COCO/H36M)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # brazo derecho
    (0, 5), (5, 6), (6, 7), (7, 8),      # brazo izquierdo
    (0, 9), (9, 10), (10, 11),           # pierna derecha
    (0, 12), (12, 13), (13, 14),         # pierna izquierda
    (0, 15), (15, 16), (16, 17)          # cabeza/cuello
]

parser = argparse.ArgumentParser(description='Visualización 3D de pose')
parser.add_argument('--npz', type=str, default='output_3d_coords.npz', help='Archivo npz con pose3d')
args = parser.parse_args()

data = np.load(args.npz)

# Verificar qué keys tiene el archivo
available_keys = list(data.keys())
print(f"📦 Keys disponibles: {available_keys}")

# Determinar el formato del archivo
if 'pose3d' in available_keys:
    pose3d = data['pose3d']  # Formato individual
    print(f"📊 Formato individual - Datos cargados: {pose3d.shape}")
elif 'poses' in available_keys:
    pose3d = data['poses']  # Formato múltiple
    print(f"📊 Formato múltiple - Datos cargados: {pose3d.shape}")
    print(f"� Depths: {data['depths'].shape}")
    print(f"📊 Frames: {data['frames'].shape}")
else:
    print(f"❌ No se encontró 'pose3d' o 'poses' en {available_keys}")
    exit(1)

# Si tenemos múltiples frames, mostrar solo algunos
if len(pose3d.shape) == 3:  # (frames, joints, coords)
    print(f"🎬 Detectados {pose3d.shape[0]} frames")
    frames_to_show = min(6, pose3d.shape[0])
    
    for frame_idx in range(frames_to_show):
        current_pose = pose3d[frame_idx]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar joints
        ax.scatter(current_pose[:, 0], current_pose[:, 1], current_pose[:, 2], c='r', s=40)
        
        # Dibujar huesos
        for i, j in SKELETON:
            if i < current_pose.shape[0] and j < current_pose.shape[0]:
                ax.plot([current_pose[i, 0], current_pose[j, 0]],
                        [current_pose[i, 1], current_pose[j, 1]],
                        [current_pose[i, 2], current_pose[j, 2]], c='b')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (depth)')
        ax.set_title(f'Visualización 3D de Pose - Frame {frame_idx}')
        
        # Guardar la visualización
        plt.tight_layout()
        output_file = f"pose_3d_frame_{frame_idx}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📸 Visualización 3D guardada: {output_file}")
        
        plt.close()  # Cerrar para liberar memoria
    
    print(f"📊 Visualizaciones guardadas para primeros {frames_to_show} frames")

else:  # Una sola pose (18, 3)
    print("📦 Visualizando pose única")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar joints
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c='r', s=40)
    
    # Dibujar huesos
    for i, j in SKELETON:
        if i < pose3d.shape[0] and j < pose3d.shape[0]:
            ax.plot([pose3d[i, 0], pose3d[j, 0]],
                    [pose3d[i, 1], pose3d[j, 1]],
                    [pose3d[i, 2], pose3d[j, 2]], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (depth)')
    ax.set_title('Visualización 3D de Pose')
    
    # Guardar la visualización
    plt.tight_layout()
    output_file = "pose_3d_single.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📸 Visualización 3D guardada: {output_file}")
    
    plt.close()