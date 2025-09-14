#!/usr/bin/env python3

"""
Análisis Académico: Profundidad Relativa vs Absoluta
====================================================
Clase magistral sobre las diferencias entre tipos de profundidad en pose estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_depth_distribution(pose_data, title=""):
    """Analiza la distribución de profundidad en los datos"""
    print(f"\n📊 ANÁLISIS DE PROFUNDIDAD: {title}")
    print("=" * 60)
    
    # Extraer coordenadas Z (profundidad)
    z_coords = pose_data[..., 2]
    
    if len(pose_data.shape) == 3:  # Multiple poses
        z_coords = z_coords.flatten()
    
    unique_z = np.unique(z_coords)
    
    print(f"💎 Valores únicos de Z: {len(unique_z)}")
    print(f"📏 Rango Z: [{z_coords.min():.1f}, {z_coords.max():.1f}]")
    print(f"📐 Desviación estándar: {z_coords.std():.1f}")
    print(f"📊 Media: {z_coords.mean():.1f}")
    
    if len(unique_z) <= 10:
        print(f"🔍 Valores exactos: {unique_z}")
    
    # Análisis por frame si es múltiple
    if len(pose_data.shape) == 3:
        print(f"\n📈 ANÁLISIS POR FRAME:")
        for i in range(min(5, pose_data.shape[0])):
            frame_z = pose_data[i, :, 2]
            unique_frame_z = np.unique(frame_z)
            print(f"   Frame {i}: {len(unique_frame_z)} valores únicos, rango [{frame_z.min():.1f}, {frame_z.max():.1f}]")
    
    return {
        'unique_count': len(unique_z),
        'range': (z_coords.min(), z_coords.max()),
        'std': z_coords.std(),
        'mean': z_coords.mean(),
        'values': unique_z if len(unique_z) <= 10 else None
    }

def create_depth_comparison_visualization():
    """Crea visualización educativa de tipos de profundidad"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Esqueleto ejemplo simplificado (5 puntos)
    # Persona parada de frente
    person_2d = np.array([
        [100, 50],   # cabeza
        [100, 100],  # cuello/torso
        [80, 120],   # brazo izq
        [120, 120],  # brazo der
        [100, 150]   # cadera
    ])
    
    # 1. Profundidad Relativa (ConvNeXtPose)
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Profundidades relativas anatómicamente correctas
    relative_z = np.array([10, 0, 5, 5, -3])  # cabeza adelante, brazos medio, cadera atrás
    
    person_3d_relative = np.column_stack([person_2d, relative_z])
    
    colors = ['red', 'green', 'blue', 'blue', 'orange']
    for i, (point, color) in enumerate(zip(person_3d_relative, colors)):
        ax1.scatter(point[0], point[1], point[2], c=color, s=100)
        ax1.text(point[0], point[1], point[2], f'J{i}', fontsize=8)
    
    # Conectar puntos
    connections = [(0,1), (1,2), (1,3), (1,4)]
    for i, j in connections:
        ax1.plot([person_3d_relative[i,0], person_3d_relative[j,0]],
                [person_3d_relative[i,1], person_3d_relative[j,1]],
                [person_3d_relative[i,2], person_3d_relative[j,2]], 'k-', alpha=0.5)
    
    ax1.set_title('PROFUNDIDAD RELATIVA\n(ConvNeXtPose)\nProporciones anatómicas correctas')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z relativo')
    
    # 2. Sin profundidad absoluta (problema actual)
    ax2 = fig.add_subplot(222, projection='3d')
    
    # Todas las articulaciones a la misma profundidad absoluta
    absolute_z_constant = np.full(5, 800)  # 800mm de la cámara
    person_3d_flat = np.column_stack([person_2d, absolute_z_constant])
    
    for i, (point, color) in enumerate(zip(person_3d_flat, colors)):
        ax2.scatter(point[0], point[1], point[2], c=color, s=100)
        ax2.text(point[0], point[1], point[2], f'J{i}', fontsize=8)
    
    for i, j in connections:
        ax2.plot([person_3d_flat[i,0], person_3d_flat[j,0]],
                [person_3d_flat[i,1], person_3d_flat[j,1]],
                [person_3d_flat[i,2], person_3d_flat[j,2]], 'k-', alpha=0.5)
    
    ax2.set_title('SIN PROFUNDIDAD ABSOLUTA\n(Problema actual)\nTodas las articulaciones planas')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z absoluto (mm)')
    
    # 3. Con RootNet - Profundidad absoluta correcta
    ax3 = fig.add_subplot(223, projection='3d')
    
    # RootNet proporciona profundidad absoluta base
    rootnet_depth = 800  # mm desde la cámara
    # Combinamos: profundidad absoluta base + profundidad relativa
    person_3d_complete = np.column_stack([person_2d, rootnet_depth + relative_z])
    
    for i, (point, color) in enumerate(zip(person_3d_complete, colors)):
        ax3.scatter(point[0], point[1], point[2], c=color, s=100)
        ax3.text(point[0], point[1], point[2], f'J{i}', fontsize=8)
    
    for i, j in connections:
        ax3.plot([person_3d_complete[i,0], person_3d_complete[j,0]],
                [person_3d_complete[i,1], person_3d_complete[j,1]],
                [person_3d_complete[i,2], person_3d_complete[j,2]], 'k-', alpha=0.5)
    
    ax3.set_title('CON ROOTNET\n(Solución completa)\nProfundidad absoluta + relativa')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z absoluto (mm)')
    
    # 4. Múltiples personas (ventaja de RootNet)
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Persona 1 cerca (600mm)
    person1_depth = 600
    person1_3d = np.column_stack([person_2d, person1_depth + relative_z])
    
    # Persona 2 lejos (1200mm)
    person2_2d = person_2d + np.array([150, 0])  # Desplazada en X
    person2_depth = 1200
    person2_3d = np.column_stack([person2_2d, person2_depth + relative_z])
    
    # Dibujar persona 1
    for i, point in enumerate(person1_3d):
        ax4.scatter(point[0], point[1], point[2], c='red', s=100, alpha=0.7)
    
    # Dibujar persona 2
    for i, point in enumerate(person2_3d):
        ax4.scatter(point[0], point[1], point[2], c='blue', s=100, alpha=0.7)
    
    ax4.set_title('MÚLTIPLES PERSONAS\n(Ventaja principal de RootNet)\nPosicionamiento relativo correcto')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z absoluto (mm)')
    
    plt.tight_layout()
    return fig

def explain_advantages_disadvantages():
    """Explicación académica de ventajas y desventajas"""
    
    print("\n" + "="*80)
    print("🎓 ANÁLISIS ACADÉMICO: VENTAJAS Y DESVENTAJAS")
    print("="*80)
    
    print("\n📚 1. PROFUNDIDAD RELATIVA (ConvNeXtPose)")
    print("-" * 50)
    print("✅ VENTAJAS:")
    print("   • Mantiene proporciones anatómicas correctas")
    print("   • Invariante a la distancia de la cámara")
    print("   • Permite análisis de movimiento y postura")
    print("   • Eficiente computacionalmente")
    print("   • Robusto a cambios de escala")
    
    print("\n❌ DESVENTAJAS:")
    print("   • No proporciona ubicación real en el espacio")
    print("   • No permite medidas métricas absolutas")
    print("   • Insuficiente para aplicaciones de realidad aumentada")
    print("   • No resuelve oclusiones entre múltiples personas")
    
    print("\n📚 2. PROFUNDIDAD ABSOLUTA (RootNet)")
    print("-" * 50)
    print("✅ VENTAJAS:")
    print("   • Ubicación real en coordenadas del mundo")
    print("   • Permite medidas métricas (distancias, volúmenes)")
    print("   • Esencial para realidad aumentada/virtual")
    print("   • Resuelve ambigüedades entre múltiples personas")
    print("   • Permite navegación y planificación de rutas")
    
    print("\n❌ DESVENTAJAS:")
    print("   • Más complejo computacionalmente")
    print("   • Requiere calibración de cámara")
    print("   • Sensible a condiciones de iluminación")
    print("   • Puede ser menos preciso en poses complejas")
    
    print("\n📚 3. COMBINACIÓN ÓPTIMA (ConvNeXtPose + RootNet)")
    print("-" * 50)
    print("✅ VENTAJAS COMBINADAS:")
    print("   • Proporciones anatómicas + ubicación real")
    print("   • Máxima precisión en poses individuales")
    print("   • Capacidad para múltiples personas")
    print("   • Aplicable a realidad aumentada")
    print("   • Análisis biomecánico completo")

def demonstrate_use_cases():
    """Demuestra casos de uso específicos"""
    
    print("\n" + "="*80)
    print("🏥 CASOS DE USO PRÁCTICOS")
    print("="*80)
    
    print("\n💊 CASO 1: ANÁLISIS MÉDICO - Rehabilitación")
    print("-" * 50)
    print("📋 Requisito: Medir ángulos de flexión de rodilla")
    print("🔧 Solución: PROFUNDIDAD RELATIVA suficiente")
    print("💡 Razón: Los ángulos son independientes de la distancia absoluta")
    print("📊 Ejemplo: Flexión 90° es 90° independiente si el paciente está a 1m o 2m")
    
    print("\n🏃 CASO 2: ANÁLISIS DEPORTIVO - Técnica de salto")
    print("-" * 50)
    print("📋 Requisito: Altura máxima del salto en metros")
    print("🔧 Solución: PROFUNDIDAD ABSOLUTA necesaria")
    print("💡 Razón: Necesitamos medidas métricas reales")
    print("📊 Ejemplo: Salto de 0.5m vs 0.8m requiere referencia absoluta")
    
    print("\n🎮 CASO 3: REALIDAD AUMENTADA - Avatar virtual")
    print("-" * 50)
    print("📋 Requisito: Colocar objeto virtual en la mano")
    print("🔧 Solución: AMBAS profundidades necesarias")
    print("💡 Razón: Ubicación precisa en espacio 3D real")
    print("📊 Ejemplo: Mano a 0.8m de cámara + 0.1m adelante del torso")
    
    print("\n👥 CASO 4: MÚLTIPLES PERSONAS - Distancia social")
    print("-" * 50)
    print("📋 Requisito: Medir distancia entre personas")
    print("🔧 Solución: PROFUNDIDAD ABSOLUTA esencial")
    print("💡 Razón: Profundidad relativa no resuelve separación entre personas")
    print("📊 Ejemplo: Persona A a 2m, Persona B a 3m → distancia 1m")

if __name__ == "__main__":
    print("🎓 CLASE MAGISTRAL: PROFUNDIDAD EN POSE ESTIMATION")
    print("="*80)
    print("Profesor: AI Assistant")
    print("Materia: Visión por Computadora Avanzada")
    print("Tema: Análisis de Profundidad en Estimación de Poses Humanas")
    
    # Cargar y analizar datos reales
    try:
        print("\n📊 ANÁLISIS DE DATOS EXPERIMENTALES")
        print("="*50)
        
        # Analizar datos individuales
        data_single = np.load('output_3d_coords.npz')
        if 'pose3d' in data_single.keys():
            pose_single = data_single['pose3d']
            result_single = analyze_depth_distribution(pose_single, "POSE INDIVIDUAL")
        
        # Analizar datos múltiples
        data_multiple = np.load('all_3d_poses.npz')
        if 'poses' in data_multiple.keys():
            poses_multiple = data_multiple['poses']
            result_multiple = analyze_depth_distribution(poses_multiple, "MÚLTIPLES POSES")
    
    except FileNotFoundError:
        print("⚠️ Archivos de datos no encontrados. Usando ejemplos teóricos.")
    
    # Crear visualización educativa
    print("\n📈 GENERANDO VISUALIZACIÓN EDUCATIVA...")
    fig = create_depth_comparison_visualization()
    fig.savefig('depth_analysis_educational.png', dpi=150, bbox_inches='tight')
    print("📸 Visualización guardada: depth_analysis_educational.png")
    
    # Explicaciones académicas
    explain_advantages_disadvantages()
    demonstrate_use_cases()
    
    print("\n" + "="*80)
    print("🎯 CONCLUSIÓN DE LA CLASE")
    print("="*80)
    print("1. ConvNeXtPose proporciona PROFUNDIDAD RELATIVA (anatómica)")
    print("2. RootNet proporciona PROFUNDIDAD ABSOLUTA (ubicación real)")
    print("3. AMBAS son necesarias para aplicaciones completas")
    print("4. Una sola persona AÚN se beneficia de profundidad absoluta")
    print("5. La combinación es superior a cualquier enfoque individual")
    
    print("\n💡 RESPUESTA A TU PREGUNTA:")
    print("Incluso con una sola persona, RootNet proporciona:")
    print("• Ubicación real en el espacio (no solo proporciones)")
    print("• Capacidad de medidas métricas absolutas")
    print("• Compatibilidad con aplicaciones de realidad aumentada")
    print("• Base para futuras expansiones a múltiples personas")
    
    plt.close()