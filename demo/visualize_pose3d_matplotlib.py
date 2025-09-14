import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def print_pose_analysis(pose_data, name="pose"):
    """Imprime análisis detallado de los datos de pose"""
    print(f"\n📊 ANÁLISIS DE {name.upper()}:")
    print(f"   Shape: {pose_data.shape}")
    print(f"   Rango X: [{pose_data[..., 0].min():.1f}, {pose_data[..., 0].max():.1f}]")
    print(f"   Rango Y: [{pose_data[..., 1].min():.1f}, {pose_data[..., 1].max():.1f}]") 
    print(f"   Rango Z: [{pose_data[..., 2].min():.1f}, {pose_data[..., 2].max():.1f}]")
    
    # Detectar joints en (0,0,0) que pueden indicar joints no detectados
    if len(pose_data.shape) == 2:  # single pose
        zero_joints = np.where(np.all(pose_data == 0, axis=1))[0]
        if len(zero_joints) > 0:
            print(f"   ⚠️ Joints en (0,0,0): {zero_joints} - posiblemente no detectados")
    
    print(f"   Media X: {pose_data[..., 0].mean():.1f}, Y: {pose_data[..., 1].mean():.1f}, Z: {pose_data[..., 2].mean():.1f}")

# Conexiones de esqueleto para 18 joints (ConvNeXt format)
# Basado en el mapeo real usado en convnext_pose_production_final_corrected.py
SKELETON = [
    (10, 9), (9, 8), (8, 11), (8, 14),      # torso central
    (11, 12), (12, 13), (14, 15), (15, 16), # piernas
    (11, 4), (14, 1), (0, 4), (0, 1),       # conexiones torso-brazos
    (4, 5), (5, 6), (1, 2), (2, 3)          # brazos
]

# Nombres de articulaciones para referencia (ConvNeXt/COCO format)
JOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1  
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
    'neck'            # 17
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
    print_pose_analysis(pose3d, "múltiples frames")
    
    frames_to_show = min(6, pose3d.shape[0])
    
    for frame_idx in range(frames_to_show):
        current_pose = pose3d[frame_idx]
        print(f"\n🖼️ Procesando frame {frame_idx}")
        print_pose_analysis(current_pose, f"frame {frame_idx}")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar joints con diferentes colores para diferentes partes del cuerpo
        # Cabeza (rojo)
        head_joints = [0, 1, 2, 3, 4]  # nose, eyes, ears
        # Torso (verde)
        torso_joints = [5, 6, 11, 12, 17]  # shoulders, hips, neck
        # Brazos (azul)
        arm_joints = [7, 8, 9, 10]  # elbows, wrists
        # Piernas (amarillo)
        leg_joints = [13, 14, 15, 16]  # knees, ankles
        
        for i in range(current_pose.shape[0]):
            if i in head_joints:
                color = 'red'
                size = 50
            elif i in torso_joints:
                color = 'green'
                size = 60
            elif i in arm_joints:
                color = 'blue'
                size = 40
            elif i in leg_joints:
                color = 'orange'
                size = 40
            else:
                color = 'gray'
                size = 30
            
            # Usar el mismo sistema de coordenadas que vis.py: (X, Z, -Y)
            ax.scatter(current_pose[i, 0], current_pose[i, 2], -current_pose[i, 1], 
                      c=color, s=size, alpha=0.8)
            
            # Agregar etiquetas a algunos joints importantes
            if i in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
                ax.text(current_pose[i, 0], current_pose[i, 2], -current_pose[i, 1], 
                       f'{i}', fontsize=8, alpha=0.7)
        
        # Dibujar huesos con colores diferentes
        for i, j in SKELETON:
            if i < current_pose.shape[0] and j < current_pose.shape[0]:
                # Determinar color de la conexión
                if (i in head_joints and j in head_joints):
                    line_color = 'red'
                elif (i in torso_joints and j in torso_joints) or \
                     (i in torso_joints and j in head_joints) or \
                     (i in head_joints and j in torso_joints):
                    line_color = 'green'
                elif (i in arm_joints and j in arm_joints) or \
                     (i in arm_joints and j in torso_joints) or \
                     (i in torso_joints and j in arm_joints):
                    line_color = 'blue'
                elif (i in leg_joints and j in leg_joints) or \
                     (i in leg_joints and j in torso_joints) or \
                     (i in torso_joints and j in leg_joints):
                    line_color = 'orange'
                else:
                    line_color = 'gray'
                
                # Usar el mismo sistema de coordenadas que vis.py: (X, Z, -Y)
                ax.plot([current_pose[i, 0], current_pose[j, 0]],
                        [current_pose[i, 2], current_pose[j, 2]],
                        [-current_pose[i, 1], -current_pose[j, 1]], 
                        c=line_color, linewidth=2, alpha=0.7)
        
        # Etiquetas de ejes corregidas para coincidir con vis.py
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label') 
        ax.set_zlabel('Y Label')
        ax.set_title(f'ConvNeXt Pose 3D - Frame {frame_idx}\nRojo=Cabeza, Verde=Torso, Azul=Brazos, Naranja=Piernas')
        
        # Ajustar vista para mejor visualización
        ax.view_init(elev=20, azim=45)
        
        # Guardar la visualización
        plt.tight_layout()
        output_file = f"pose_3d_frame_{frame_idx}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📸 Visualización 3D guardada: {output_file}")
        
        plt.close()  # Cerrar para liberar memoria
    
    print(f"📊 Visualizaciones guardadas para primeros {frames_to_show} frames")

else:  # Una sola pose (18, 3)
    print("📦 Visualizando pose única")
    print_pose_analysis(pose3d, "pose única")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar joints con diferentes colores para diferentes partes del cuerpo
    # Cabeza (rojo)
    head_joints = [0, 1, 2, 3, 4]  # nose, eyes, ears
    # Torso (verde)
    torso_joints = [5, 6, 11, 12, 17]  # shoulders, hips, neck
    # Brazos (azul)
    arm_joints = [7, 8, 9, 10]  # elbows, wrists
    # Piernas (amarillo)
    leg_joints = [13, 14, 15, 16]  # knees, ankles
    
    for i in range(pose3d.shape[0]):
        if i in head_joints:
            color = 'red'
            size = 50
        elif i in torso_joints:
            color = 'green'
            size = 60
        elif i in arm_joints:
            color = 'blue'
            size = 40
        elif i in leg_joints:
            color = 'orange'
            size = 40
        else:
            color = 'gray'
            size = 30
        
        # Usar el mismo sistema de coordenadas que vis.py: (X, Z, -Y)
        ax.scatter(pose3d[i, 0], pose3d[i, 2], -pose3d[i, 1], 
                  c=color, s=size, alpha=0.8)
        
        # Agregar etiquetas a algunos joints importantes
        if i in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
            ax.text(pose3d[i, 0], pose3d[i, 2], -pose3d[i, 1], 
                   f'{i}:{JOINT_NAMES[i][:4]}', fontsize=8, alpha=0.7)
    
    # Dibujar huesos con colores diferentes
    for i, j in SKELETON:
        if i < pose3d.shape[0] and j < pose3d.shape[0]:
            # Determinar color de la conexión
            if (i in head_joints and j in head_joints):
                line_color = 'red'
            elif (i in torso_joints and j in torso_joints) or \
                 (i in torso_joints and j in head_joints) or \
                 (i in head_joints and j in torso_joints):
                line_color = 'green'
            elif (i in arm_joints and j in arm_joints) or \
                 (i in arm_joints and j in torso_joints) or \
                 (i in torso_joints and j in arm_joints):
                line_color = 'blue'
            elif (i in leg_joints and j in leg_joints) or \
                 (i in leg_joints and j in torso_joints) or \
                 (i in torso_joints and j in leg_joints):
                line_color = 'orange'
            else:
                line_color = 'gray'
            
            # Usar el mismo sistema de coordenadas que vis.py: (X, Z, -Y)
            ax.plot([pose3d[i, 0], pose3d[j, 0]],
                    [pose3d[i, 2], pose3d[j, 2]],
                    [-pose3d[i, 1], -pose3d[j, 1]], 
                    c=line_color, linewidth=2, alpha=0.7)

    # Etiquetas de ejes corregidas para coincidir con vis.py
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_title('ConvNeXt Pose 3D\nRojo=Cabeza, Verde=Torso, Azul=Brazos, Naranja=Piernas')
    
    # Ajustar vista para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    # Guardar la visualización
    plt.tight_layout()
    output_file = "pose_3d_single.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📸 Visualización 3D guardada: {output_file}")
    
    plt.close()