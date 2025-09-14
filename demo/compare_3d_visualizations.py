#!/usr/bin/env python3

"""
Comparador de visualizaciones 3D - Original vs Corregido
========================================================
Muestra lado a lado las poses con el sistema de coordenadas original (X,Y,Z) 
vs el sistema corregido del demo.py (X,Z,-Y)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Conexiones de esqueleto para 18 joints (ConvNeXt format)
SKELETON = [
    (10, 9), (9, 8), (8, 11), (8, 14),      # torso central
    (11, 12), (12, 13), (14, 15), (15, 16), # piernas
    (11, 4), (14, 1), (0, 4), (0, 1),       # conexiones torso-brazos
    (4, 5), (5, 6), (1, 2), (2, 3)          # brazos
]

# Nombres de articulaciones para referencia
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'neck'
]

def plot_pose_comparison(pose_data, title_prefix=""):
    """Muestra comparación lado a lado de sistemas de coordenadas"""
    
    fig = plt.figure(figsize=(20, 10))
    
    # Configuración de colores para joints
    head_joints = [0, 1, 2, 3, 4]
    torso_joints = [5, 6, 11, 12, 17]
    arm_joints = [7, 8, 9, 10]
    leg_joints = [13, 14, 15, 16]
    
    # Plot 1: Sistema original (X, Y, Z)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f'{title_prefix} - Sistema Original (X,Y,Z)\n⚠️ Puede verse plano o incorrecto')
    
    for i in range(pose_data.shape[0]):
        if i in head_joints:
            color = 'red'
        elif i in torso_joints:
            color = 'green'
        elif i in arm_joints:
            color = 'blue'
        elif i in leg_joints:
            color = 'orange'
        else:
            color = 'gray'
        
        ax1.scatter(pose_data[i, 0], pose_data[i, 1], pose_data[i, 2], 
                   c=color, s=50, alpha=0.8)
        
        # Etiquetas para joints importantes
        if i in [0, 5, 6, 11, 12]:
            ax1.text(pose_data[i, 0], pose_data[i, 1], pose_data[i, 2], 
                    f'{i}', fontsize=6, alpha=0.7)
    
    # Dibujar skeleton original
    for i, j in SKELETON:
        if i < pose_data.shape[0] and j < pose_data.shape[0]:
            ax1.plot([pose_data[i, 0], pose_data[j, 0]],
                    [pose_data[i, 1], pose_data[j, 1]],
                    [pose_data[i, 2], pose_data[j, 2]], 
                    c='gray', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_zlabel('Z (depth)')
    ax1.view_init(elev=20, azim=45)
    
    # Plot 2: Sistema corregido (X, Z, -Y) como en demo.py
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f'{title_prefix} - Sistema Demo.py (X,Z,-Y)\n✅ Visualización corregida')
    
    for i in range(pose_data.shape[0]):
        if i in head_joints:
            color = 'red'
        elif i in torso_joints:
            color = 'green'
        elif i in arm_joints:
            color = 'blue'
        elif i in leg_joints:
            color = 'orange'
        else:
            color = 'gray'
        
        ax2.scatter(pose_data[i, 0], pose_data[i, 2], -pose_data[i, 1], 
                   c=color, s=50, alpha=0.8)
        
        # Etiquetas para joints importantes
        if i in [0, 5, 6, 11, 12]:
            ax2.text(pose_data[i, 0], pose_data[i, 2], -pose_data[i, 1], 
                    f'{i}:{JOINT_NAMES[i][:4]}', fontsize=6, alpha=0.7)
    
    # Dibujar skeleton corregido
    for i, j in SKELETON:
        if i < pose_data.shape[0] and j < pose_data.shape[0]:
            ax2.plot([pose_data[i, 0], pose_data[j, 0]],
                    [pose_data[i, 2], pose_data[j, 2]],
                    [-pose_data[i, 1], -pose_data[j, 1]], 
                    c='gray', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Z Label')
    ax2.set_zlabel('Y Label')
    ax2.view_init(elev=20, azim=45)
    
    # Añadir leyenda
    fig.suptitle('Comparación de Sistemas de Coordenadas 3D\nRojo=Cabeza, Verde=Torso, Azul=Brazos, Naranja=Piernas', 
                fontsize=16, y=0.95)
    
    plt.tight_layout()
    return fig

def print_detailed_analysis(pose_data, name="pose"):
    """Análisis detallado de la pose"""
    print(f"\n🔍 ANÁLISIS DETALLADO DE {name.upper()}:")
    print(f"   📐 Shape: {pose_data.shape}")
    print(f"   📊 Estadísticas por eje:")
    print(f"      X: min={pose_data[..., 0].min():.1f}, max={pose_data[..., 0].max():.1f}, mean={pose_data[..., 0].mean():.1f}")
    print(f"      Y: min={pose_data[..., 1].min():.1f}, max={pose_data[..., 1].max():.1f}, mean={pose_data[..., 1].mean():.1f}")
    print(f"      Z: min={pose_data[..., 2].min():.1f}, max={pose_data[..., 2].max():.1f}, mean={pose_data[..., 2].mean():.1f}")
    
    # Análisis de variabilidad en Z (profundidad)
    z_values = pose_data[..., 2].flatten()
    unique_z = np.unique(z_values)
    print(f"   🎯 Profundidad (Z): {len(unique_z)} valores únicos")
    
    if len(unique_z) == 1:
        print(f"      ⚠️ PROBLEMA: Todas las articulaciones tienen la misma profundidad Z={unique_z[0]:.1f}")
        print(f"         Esto indica que no hay información de profundidad real entre articulaciones")
    else:
        print(f"      ✅ Variabilidad en Z: rango de {z_values.max() - z_values.min():.1f}mm")
    
    # Verificar joints en (0,0,0)
    if len(pose_data.shape) == 2:  # single pose
        zero_joints = np.where(np.all(pose_data == 0, axis=1))[0]
        if len(zero_joints) > 0:
            print(f"   ⚠️ Joints no detectados (0,0,0): {zero_joints}")
            print(f"      Nombres: {[JOINT_NAMES[i] for i in zero_joints]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparación de visualizaciones 3D')
    parser.add_argument('--npz', type=str, default='output_3d_coords.npz', 
                       help='Archivo npz con pose3d')
    parser.add_argument('--frame', type=int, default=0, 
                       help='Frame a visualizar (para archivos múltiples)')
    args = parser.parse_args()
    
    print("🔄 COMPARADOR DE VISUALIZACIONES 3D")
    print("=" * 50)
    
    data = np.load(args.npz)
    available_keys = list(data.keys())
    print(f"📦 Keys disponibles: {available_keys}")
    
    # Cargar datos
    if 'pose3d' in available_keys:
        pose3d = data['pose3d']
        print(f"📊 Formato individual - Shape: {pose3d.shape}")
        print_detailed_analysis(pose3d, "pose única")
        
        fig = plot_pose_comparison(pose3d, "Pose Única")
        output_file = "comparison_single_pose.png"
        
    elif 'poses' in available_keys:
        poses = data['poses']
        print(f"📊 Formato múltiple - Shape: {poses.shape}")
        
        frame_idx = min(args.frame, poses.shape[0] - 1)
        pose3d = poses[frame_idx]
        
        print(f"🖼️ Visualizando frame {frame_idx}")
        print_detailed_analysis(pose3d, f"frame {frame_idx}")
        
        fig = plot_pose_comparison(pose3d, f"Frame {frame_idx}")
        output_file = f"comparison_frame_{frame_idx}.png"
        
    else:
        print(f"❌ No se encontró 'pose3d' o 'poses' en {available_keys}")
        exit(1)
    
    # Guardar comparación
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📸 Comparación guardada: {output_file}")
    print("\n💡 CONCLUSIÓN:")
    print("   - Lado izquierdo: Sistema original que puede verse plano")
    print("   - Lado derecho: Sistema del demo.py que debería verse más natural")
    print("   - El sistema corregido usa (X, Z, -Y) en lugar de (X, Y, Z)")
    
    plt.close()