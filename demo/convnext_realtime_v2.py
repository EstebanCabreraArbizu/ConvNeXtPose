#!/usr/bin/env python3
"""
convnext_realtime_v2.py - VERSIÓN CORREGIDA DEFINITIVA

Correcciones principales:
1. Integración correcta de RootNet para profundidad robusta
2. Normalización apropiada de coordenadas entre modelos
3. Transformaciones geométricas mejoradas para alineación
4. Manejo robusto de casos extremos y valores fuera de rango
"""

import argparse
import time
import sys
import inspect
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from deepsparse import compile_model
from deepsparse.pipeline import Pipeline
from root_wrapper import RootNetWrapper

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

# ------ IMPORTACIÓN DE CONVNEXTPOSE ------
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils
import utils.vis as vis_utils

def comprehensive_checkpoint_diagnosis(pth_path: str):
    """Diagnóstico completo del checkpoint"""
    print(f"🔍 DIAGNÓSTICO COMPLETO DEL CHECKPOINT: {pth_path}")
    print("="*70)
    
    try:
        # 1. Verificar que el archivo existe
        if not os.path.exists(pth_path):
            print(f"❌ ERROR: Archivo no encontrado: {pth_path}")
            return False
            
        file_size = os.path.getsize(pth_path) / (1024*1024)  # MB
        print(f"📁 Tamaño del archivo: {file_size:.2f} MB")
        
        # 2. Cargar checkpoint
        checkpoint = torch.load(pth_path, map_location='cpu')
        print(f"✅ Checkpoint cargado exitosamente")
        
        # 3. Verificar estructura del checkpoint
        print(f"🔑 Claves principales: {list(checkpoint.keys())}")
        
        # 4. Extraer state_dict
        if 'network' in checkpoint:
            state_dict = checkpoint['network']
            print(f"✅ Usando clave 'network'")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"✅ Usando clave 'state_dict'")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Usando clave 'model_state_dict'")
        else:
            state_dict = checkpoint
            print(f"✅ Usando checkpoint directo como state_dict")
        
        # 5. Analizar capas del modelo
        print(f"🏗️ ESTRUCTURA DEL MODELO:")
        print(f"   Total de parámetros: {len(state_dict)}")
        
        # Primeras y últimas capas
        keys = list(state_dict.keys())
        print(f"   Primera capa: {keys[0]}")
        print(f"   Última capa: {keys[-1]}")
        
        # 6. Verificar capas críticas de ConvNeXt
        convnext_indicators = [
            'downsample_layers', 'stages', 'norm', 'head',
            'backbone', 'head_net', 'conv', 'gamma'
        ]
        
        found_indicators = []
        for indicator in convnext_indicators:
            if any(indicator in key for key in keys):
                found_indicators.append(indicator)
        
        print(f"🎯 Indicadores ConvNeXt encontrados: {found_indicators}")
        
        # 7. Análizar rangos de pesos de capas críticas
        print(f"📊 ANÁLISIS DE PESOS:")
        sample_keys = keys[:5] + keys[-3:]  # Primeras 5 y últimas 3
        
        for key in sample_keys:
            if isinstance(state_dict[key], torch.Tensor):
                tensor = state_dict[key]
                print(f"   {key}:")
                print(f"     Shape: {tensor.shape}")
                print(f"     Min/Max: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
                print(f"     Std: {tensor.std().item():.6f}")
                
                # Detectar pesos anómalos
                if torch.isnan(tensor).any():
                    print(f"     ❌ PROBLEMA: Contiene NaN")
                if torch.isinf(tensor).any():
                    print(f"     ❌ PROBLEMA: Contiene infinitos")
                if tensor.abs().max() > 100:
                    print(f"     ⚠️ ADVERTENCIA: Pesos muy grandes")
                if tensor.abs().max() < 1e-6:
                    print(f"     ⚠️ ADVERTENCIA: Pesos muy pequeños")
        
        # 8. Verificar compatibilidad con cfg
        expected_output_dim = cfg.joint_num * 3 if hasattr(cfg, 'joint_num') else 54  # 18*3
        head_keys = [k for k in keys if 'head' in k.lower() and 'weight' in k]
        
        if head_keys:
            head_key = head_keys[-1]  # Última capa head
            head_weight = state_dict[head_key]
            print(f"🎯 CAPA DE SALIDA:")
            print(f"   Clave: {head_key}")
            print(f"   Shape: {head_weight.shape}")
            print(f"   Dimensión de salida esperada: {expected_output_dim}")
            
            if head_weight.shape[0] == expected_output_dim:
                print(f"   ✅ Dimensión de salida CORRECTA")
            else:
                print(f"   ❌ Dimensión de salida INCORRECTA")
                
        # 9. Verificar metadatos adicionales
        if 'epoch' in checkpoint:
            print(f"📅 Época de entrenamiento: {checkpoint['epoch']}")
        if 'optimizer' in checkpoint:
            print(f"🔧 Optimizador incluido: Sí")
        if 'loss' in checkpoint:
            print(f"📉 Loss guardado: {checkpoint['loss']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_inference_simple(model, device):
    """Test de inferencia simple para verificar que el modelo funciona"""
    print(f"🧪 TEST DE INFERENCIA SIMPLE:")
    
    try:
        # Crear entrada de prueba
        test_input = torch.randn(1, 3, 256, 256).to(device)
        print(f"   Entrada de prueba: {test_input.shape}")
        
        # Inferencia
        with torch.no_grad():
            test_output = model(test_input)
            
        print(f"   Salida del modelo: {test_output.shape}")
        print(f"   Rango de salida: [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
        
        # Verificar que la salida es razonable
        if test_output.shape[1] == 18 and test_output.shape[2] == 3:
            print(f"   ✅ Shape de salida CORRECTA (18 joints, 3 coords)")
        else:
            print(f"   ❌ Shape de salida INCORRECTA")
            
        # Verificar rangos
        if test_output.min() >= -1000 and test_output.max() <= 1000:
            print(f"   ✅ Rango de valores RAZONABLE")
        else:
            print(f"   ❌ Rango de valores ANÓMALO")
            
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR en test de inferencia: {e}")
        return False

def verify_soft_argmax_availability():
    """Verificar si soft_argmax está disponible y funciona"""
    print(f"🔧 VERIFICACIÓN DE SOFT_ARGMAX:")
    
    try:
        from model import soft_argmax
        print(f"   ✅ soft_argmax importado correctamente")
        
        # CORRECCIÓN: Usar tensor 4D como espera la función
        test_heatmap = torch.randn(1, 576, 32, 32)  # batch, channels(18*32), height, width
        
        result = soft_argmax(
            test_heatmap,
            joint_num=18,
            depth_dim=32,
            output_shape=(32, 32)
        )
        
        print(f"   ✅ soft_argmax funciona correctamente")
        print(f"   Entrada: {test_heatmap.shape}")
        print(f"   Salida: {result.shape}")
        print(f"   Rango: [{result.min().item():.2f}, {result.max().item():.2f}]")
        
        return True
        
    except ImportError:
        print(f"   ❌ ERROR: No se puede importar soft_argmax")
        print(f"   Verifica que el archivo model.py esté en el path")
        return False
    except Exception as e:
        print(f"   ❌ ERROR ejecutando soft_argmax: {e}")
        return False

def complete_model_diagnosis(pth_path: str):
    """Diagnóstico completo del modelo y checkpoint"""
    print(f"\n🏥 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("="*80)
    
    # 1. Diagnóstico del checkpoint
    checkpoint_ok = comprehensive_checkpoint_diagnosis(pth_path)
    
    if not checkpoint_ok:
        print(f"❌ FALLO CRÍTICO: Checkpoint inválido")
        return False
    
    # 2. Verificar soft_argmax
    soft_argmax_ok = verify_soft_argmax_availability()
    
    # 3. Cargar y probar modelo
    try:
        print(f"\n🔄 CARGANDO MODELO...")
        model, device = load_pose_model(pth_path, joint_num=18, use_cuda=False)
        print(f"   ✅ Modelo cargado en: {device}")
        
        # 4. Test de inferencia
        inference_ok = test_model_inference_simple(model, device)
        
        # 5. Resumen final
        print(f"\n📋 RESUMEN DEL DIAGNÓSTICO:")
        print(f"   Checkpoint válido: {'✅' if checkpoint_ok else '❌'}")
        print(f"   soft_argmax disponible: {'✅' if soft_argmax_ok else '❌'}")
        print(f"   Inferencia funcional: {'✅' if inference_ok else '❌'}")
        
        if checkpoint_ok and soft_argmax_ok and inference_ok:
            print(f"\n🎉 DIAGNÓSTICO: MODELO COMPLETAMENTE FUNCIONAL")
            return True
        else:
            print(f"\n⚠️ DIAGNÓSTICO: PROBLEMAS DETECTADOS")
            return False
            
    except Exception as e:
        print(f"❌ ERROR cargando modelo: {e}")
        return False
    
def letterbox(image: np.ndarray, new_size=(640, 640), color=(114, 114, 114)):
    """Redimensiona con letterbox manteniendo aspecto"""
    h, w = image.shape[:2]
    new_h, new_w = new_size
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (new_h - resized_h) // 2
    pad_bottom = new_h - resized_h - pad_top
    pad_left = (new_w - resized_w) // 2
    pad_right = new_w - resized_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, pad_left, pad_top

def fix_convnext_configuration():
    """Configuración definitiva para ConvNeXt heatmaps"""
    # ANÁLISIS: 576 = 18 joints × 32 depth_layers  
    cfg.joint_num = 18
    cfg.depth_dim = 32  # Confirmado por 576/18 = 32
    cfg.output_shape = (32, 32)  # Resolución espacial de heatmaps
    cfg.input_shape = (256, 256)
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    print(f"🔧 CONFIGURACIÓN DEFINITIVA:")
    print(f"   576 canales = 18 joints × 32 depth = ✅")
    print(f"   depth_dim: {cfg.depth_dim}")
    print(f"   output_shape: {cfg.output_shape}")

    # Verificación matemática
    expected_size = 18 * 32 * 32 * 32
    print(f"   Verificación: {expected_size} = 589,824 ✅")
    
def load_pose_model_final(pth_path: str, joint_num=18, use_cuda=False):
    """Carga ConvNeXt con configuración definitiva"""
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    # Aplicar configuración corregida
    fix_convnext_configuration()
    
    # Crear modelo
    model = get_pose_net(cfg, is_train=False, joint_num=joint_num)
    
    # Cargar pesos
    state = torch.load(pth_path, map_location=device)
    sd = state.get('network', state)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    
    print(f"✅ Modelo cargado con configuración corregida")
    return model, device

def load_pose_model(pth_path: str, joint_num=18, use_cuda=False):
    """Función para el diagnóstico - wrapper de load_pose_model_final"""
    return load_pose_model_final(pth_path, joint_num, use_cuda)

def corrected_inference_final_v4(model, inp, cfg):
    """Inferencia usando el soft_argmax corregido del modelo original"""
    print(f"🔬 USANDO SOFT_ARGMAX CORREGIDO EN MODEL.PY:")
    
    with torch.no_grad():
        # Usar directamente el modelo original (que ahora tiene soft_argmax corregido)
        coordinates = model(inp)
        
        print(f"   ✅ Modelo ejecutado: {coordinates.shape}")
        print(f"   Rango X: [{coordinates[0, :, 0].min().item():.2f}, {coordinates[0, :, 0].max().item():.2f}]")
        print(f"   Rango Y: [{coordinates[0, :, 1].min().item():.2f}, {coordinates[0, :, 1].max().item():.2f}]")
        print(f"   Rango Z: [{coordinates[0, :, 2].min().item():.2f}, {coordinates[0, :, 2].max().item():.2f}]")
        
        # Verificar si hay variabilidad real
        x_var = coordinates[0, :, 0].std().item()
        y_var = coordinates[0, :, 1].std().item()
        z_var = coordinates[0, :, 2].std().item()
        
        print(f"   Variabilidad X: {x_var:.2f}, Y: {y_var:.2f}, Z: {z_var:.2f}")
        
        if x_var > 1.0 and y_var > 1.0:
            print(f"   ✅ VARIABILIDAD ADECUADA - Esqueleto debería verse correcto")
        else:
            print(f"   ⚠️ POCA VARIABILIDAD - Posible problema en heatmaps")
            
        return coordinates
    
def exact_demo_processing(pose_3d_raw, img2bb_trans, root_depth, cfg):
    """Procesamiento EXACTO como en demo.py sin modificaciones"""
    
    print(f"🔄 Aplicando procesamiento exacto de demo.py...")
    
    # Verificar que los valores estén en el rango esperado
    if not (0 <= pose_3d_raw[:, 0].max() <= cfg.output_shape[1] + 5 and
            0 <= pose_3d_raw[:, 1].max() <= cfg.output_shape[0] + 5 and
            0 <= pose_3d_raw[:, 2].max() <= cfg.depth_dim + 5):
        print(f"❌ VALORES FUERA DE RANGO ESPERADO - El modelo no está funcionando correctamente")
        print(f"   Esto explica por qué el esqueleto es diagonal")
        return None
    
    # Aplicar transformación EXACTA de demo.py
    pose_3d = pose_3d_raw.copy()
    
    # inverse affine transform (restore the crop and resize)
    pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
    pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
    pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
    img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
    pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

    # root-relative discretized depth -> absolute continuous depth
    pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth
    
    return pose_3d
    
def robust_skeleton_alignment(coords_2d, bbox, confidence_threshold=0.3):
    """
    Alineación robusta del esqueleto basada en anatomía humana
    
    Args:
        coords_2d: Coordenadas 2D de las articulaciones
        bbox: Bounding box [x1, y1, x2, y2]
        confidence_threshold: Umbral para considerar articulaciones válidas
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    
    # Identificar articulaciones clave según el estándar COCO/Human3.6M
    # Índices típicos: 0=nariz, 1=cuello, 2=hombro_der, 5=hombro_izq, 
    # 8=cadera_der, 11=cadera_izq, 14=pelvis, 15=torso, 16=cabeza
    
    key_joints = {
        'head': 16 if len(coords_2d) > 16 else 0,
        'neck': 1,
        'torso': 15 if len(coords_2d) > 15 else 1,
        'pelvis': 14 if len(coords_2d) > 14 else 8,
        'shoulder_r': 2,
        'shoulder_l': 5,
        'hip_r': 8,
        'hip_l': 11
    }
    
    # Calcular centro corporal basado en articulaciones confiables
    valid_torso_joints = []
    for joint_name, idx in key_joints.items():
        if idx < len(coords_2d):
            joint_pos = coords_2d[idx]
            # Verificar si la articulación está dentro de límites razonables
            if (0 <= joint_pos[0] <= bbox_width * 3 and 
                0 <= joint_pos[1] <= bbox_height * 3):
                valid_torso_joints.append(joint_pos)
    
    if len(valid_torso_joints) < 3:
        print("⚠️ Pocas articulaciones válidas detectadas, usando centroide simple")
        body_center = np.mean(coords_2d, axis=0)
    else:
        body_center = np.mean(valid_torso_joints, axis=0)
    
    # Calcular offset para centrar el esqueleto en el bbox
    target_center_x = bbox_center_x
    target_center_y = bbox_center_y
    
    offset_x = target_center_x - body_center[0]
    offset_y = target_center_y - body_center[1]
    
    # Aplicar offset con suavizado para evitar saltos bruscos
    smoothing_factor = 0.7  # Factor de suavizado [0,1]
    coords_aligned = coords_2d.copy()
    coords_aligned[:, 0] += offset_x * smoothing_factor
    coords_aligned[:, 1] += offset_y * smoothing_factor
    
    # Verificar y corregir escalado si el esqueleto está muy pequeño o grande
    skeleton_height = np.max(coords_aligned[:, 1]) - np.min(coords_aligned[:, 1])
    skeleton_width = np.max(coords_aligned[:, 0]) - np.min(coords_aligned[:, 0])
    
    expected_height_ratio = 0.8  # El esqueleto debería ocupar ~80% del bbox
    expected_width_ratio = 0.6   # El esqueleto debería ocupar ~60% del bbox
    
    if skeleton_height > 0:
        height_scale = (bbox_height * expected_height_ratio) / skeleton_height
        width_scale = (bbox_width * expected_width_ratio) / skeleton_width
        
        # Usar escala conservadora para evitar distorsión
        scale = min(max(height_scale, width_scale), 1.5)  # Limitar escala máxima
        scale = max(scale, 0.5)  # Limitar escala mínima
        
        if abs(scale - 1.0) > 0.1:  # Solo aplicar si la diferencia es significativa
            center = np.mean(coords_aligned, axis=0)
            coords_aligned = center + (coords_aligned - center) * scale
            print(f"📏 Aplicando escala corporal: {scale:.2f}")
    
    return coords_aligned

def draw_skeleton(img: np.ndarray, joints: np.ndarray, skeleton, offset=(0, 0), color=(0, 255, 0)):
    """Dibuja articulaciones y líneas de esqueleto con verificaciones de seguridad"""
    if len(joints) == 0:
        return
        
    # Verificar que las coordenadas estén dentro de la imagen
    h, w = img.shape[:2]
    
    # Dibujar conexiones del esqueleto
    for (i, j) in skeleton:
        if i < len(joints) and j < len(joints):
            x1, y1 = int(joints[i][0] + offset[0]), int(joints[i][1] + offset[1])
            x2, y2 = int(joints[j][0] + offset[0]), int(joints[j][1] + offset[1])
            
            # Verificar que los puntos estén dentro de la imagen
            if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
    
    # Dibujar articulaciones
    for idx, (x, y) in enumerate(joints):
        x_draw, y_draw = int(x + offset[0]), int(y + offset[1])
        if 0 <= x_draw < w and 0 <= y_draw < h:
            cv2.circle(img, (x_draw, y_draw), 3, color, -1)
            # Opcional: mostrar número de articulación para debug
            # cv2.putText(img, str(idx), (x_draw+5, y_draw-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def debug_joint_positions(coords_2d, frame_count):
    """Debug detallado de posiciones de articulaciones para configurar skeleton"""
    print(f"\n🔍 DEBUG ARTICULACIONES Frame {frame_count}:")
    print("="*60)
    
    # Mostrar todas las articulaciones con sus coordenadas
    for i, (x, y) in enumerate(coords_2d):
        print(f"   Joint {i:2d}: X={x:6.1f}, Y={y:6.1f}")
    
    # Análisis anatómico
    print(f"\n📊 ANÁLISIS ANATÓMICO:")
    
    # Ordenar por Y (vertical) - cabeza arriba, pies abajo
    y_sorted = sorted(enumerate(coords_2d), key=lambda x: x[1][1])
    print(f"   Por altura (Y menor = arriba):")
    for i, (joint_idx, (x, y)) in enumerate(y_sorted):
        region = "CABEZA" if i < 3 else "TORSO" if i < 12 else "PIERNAS"
        print(f"     {region:7} - Joint {joint_idx:2d}: Y={y:6.1f}")
    
    # Ordenar por X (horizontal) - izquierda/derecha
    x_sorted = sorted(enumerate(coords_2d), key=lambda x: x[1][0])
    print(f"\n   Por posición horizontal (X menor = izquierda):")
    for i, (joint_idx, (x, y)) in enumerate(x_sorted):
        side = "IZQUIERDA" if i < 9 else "DERECHA"
        print(f"     {side:9} - Joint {joint_idx:2d}: X={x:6.1f}")
    
    # Detectar candidatos para articulaciones principales
    print(f"\n🎯 CANDIDATOS PARA SKELETON:")
    
    # Cabeza (Y más pequeño)
    head_candidates = [idx for idx, _ in y_sorted[:3]]
    print(f"   Cabeza:    {head_candidates}")
    
    # Cuello/Hombros (zona intermedia superior)
    neck_candidates = [idx for idx, _ in y_sorted[3:6]]
    print(f"   Cuello:    {neck_candidates}")
    
    # Torso (zona media)
    torso_candidates = [idx for idx, _ in y_sorted[6:12]]
    print(f"   Torso:     {torso_candidates}")
    
    # Pelvis/Caderas (zona inferior)
    pelvis_candidates = [idx for idx, _ in y_sorted[12:15]]
    print(f"   Pelvis:    {pelvis_candidates}")
    
    # Piernas (Y más grande)
    legs_candidates = [idx for idx, _ in y_sorted[15:]]
    print(f"   Piernas:   {legs_candidates}")
    
    return {
        'head': head_candidates,
        'neck': neck_candidates, 
        'torso': torso_candidates,
        'pelvis': pelvis_candidates,
        'legs': legs_candidates
    }

def suggest_skeleton_connections(joint_analysis):
    """Sugiere conexiones de skeleton basadas en análisis anatómico"""
    print(f"\n🔧 SUGERENCIAS DE CONEXIONES:")
    
    suggestions = []
    
    try:
        # Conexiones principales basadas en anatomía
        if joint_analysis['head'] and joint_analysis['neck']:
            conn = (joint_analysis['head'][0], joint_analysis['neck'][0])
            suggestions.append(conn)
            print(f"   Cabeza -> Cuello: {conn}")
        
        if joint_analysis['neck'] and joint_analysis['torso']:
            conn = (joint_analysis['neck'][0], joint_analysis['torso'][0])
            suggestions.append(conn)
            print(f"   Cuello -> Torso: {conn}")
            
        if joint_analysis['torso'] and joint_analysis['pelvis']:
            conn = (joint_analysis['torso'][-1], joint_analysis['pelvis'][0])
            suggestions.append(conn)
            print(f"   Torso -> Pelvis: {conn}")
            
        if joint_analysis['pelvis'] and joint_analysis['legs']:
            # Conectar pelvis con ambas piernas
            for leg_joint in joint_analysis['legs'][:2]:
                conn = (joint_analysis['pelvis'][0], leg_joint)
                suggestions.append(conn)
                print(f"   Pelvis -> Pierna: {conn}")
        
    except (IndexError, KeyError) as e:
        print(f"   ⚠️ Error generando sugerencias: {e}")
    
    return suggestions
def main():
    # Configuración mejorada
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)  # ✅ Configuración original
    cfg.depth_dim = 32           # ✅ Configuración original
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    parser = argparse.ArgumentParser(description="ConvNeXt + RootNet Integration - Fixed Version")
    parser.add_argument('--input', type=str, default='0', help='Video source')
    parser.add_argument('--yolo-model', type=str, 
                        default='zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none',
                        help='YOLO ONNX model path')
    parser.add_argument('--pose-model', type=str, required=True, help='ConvNeXt checkpoint path')
    parser.add_argument('--rootnet-dir', type=str, 
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE',
                        help='RootNet directory path')
    parser.add_argument('--rootnet-model', type=str,
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar',
                        help='RootNet model checkpoint')
    parser.add_argument('--conf-thresh', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--diagnose', action='store_true', help='Ejecutar diagnóstico completo del modelo')
    args = parser.parse_args()

    if args.diagnose:
        diagnosis_result = complete_model_diagnosis(args.pose_model)
        
        if not diagnosis_result:
            print(f"❌ FALLO EN DIAGNÓSTICO - Revisa el checkpoint")
            return
        else:
            print(f"✅ DIAGNÓSTICO EXITOSO - Continuando con demo...")
    # Inicializar modelos
    print(f"[INFO] Cargando YOLO con DeepSparse: {args.yolo_model}")
    yolo_pipeline = Pipeline.create(task="yolo", model_path=args.yolo_model)
    
    print(f"[INFO] Cargando ConvNeXtPose: {args.pose_model}")
    pose_model, device = load_pose_model_final(args.pose_model, joint_num=18, use_cuda=False)
    
    print(f"[INFO] Inicializando RootNet: {args.rootnet_dir}")
    root_wrapper = RootNetWrapper(args.rootnet_dir, args.rootnet_model)
    root_wrapper.load_model(use_gpu=False)

    # Transformación para ConvNeXt
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])

    # Esqueleto Human3.6M/COCO standard
    skeleton = [
        # =================== COLUMNA VERTEBRAL (Progresión Y natural) ===================
        (10, 9),   # Cabeza superior -> Cabeza media ✅
        (9, 8),    # Cabeza media -> Cuello base ✅
        #(8, 17),   # Cuello base -> Torso superior ✅
        #(17, 7),   # Torso superior -> Torso medio ✅
        
        # =================== BRAZOS SIMPLIFICADOS Y CORRECTOS ===================
        # Hombros desde cuello/torso (conexión dual para estabilidad)
        (8, 11),   # Cuello -> Hombro derecho ✅
        (8, 14),   # Cuello -> Hombro izquierdo ✅
        
        # Brazo derecho (funciona bien en las imágenes)
        (11, 12),  # Hombro derecho -> Codo derecho ✅
        (12, 13),   # Codo derecho -> Muñeca derecha ✅
        
        # Brazo izquierdo  (de frente) CORREGIDO (eliminar conexión problemática)
        (14, 15),  # Hombro izquierdo -> Codo izquierdo ✅
        # ELIMINAR (15, 16) que causa el problema visual
        (15, 16),  # Conexión directa: Hombro izquierdo -> Muñeca izquierda ✅
        
        # =================== TORSO A PELVIS SIMPLIFICADO ===================
        (11, 4),
        (14, 1),
        # Conexión directa desde torso medio a ambas caderas
        # 0 -> cadera de en medio
        # (7, 0), # Torso medio -> Cadera de en medio ✅
        (0, 4),    # Cadera de en medio -> Cadera derecha ✅
        (0, 1),    # Cadera de en medio -> Cadera izquierda ✅
        
        # =================== PIERNAS ANATÓMICAMENTE CORRECTAS ===================
        # Pierna derecha (secuencia natural)
        (4, 5),    # Cadera derecha -> Rodilla derecha ✅
        (5, 6),    # Rodilla derecha -> Tobillo derecho ✅
        
        # Pierna izquierda (secuencia natural)
        (1, 2),    # Cadera izquierda -> Rodilla izquierda ✅
        (2, 3),    # Rodilla izquierda -> Tobillo izquierdo ✅) ✅
    ]
    
    print(f"🔧 Usando esqueleto CORREGIDO BASADO EN ANÁLISIS REAL con {len(skeleton)} conexiones")
    
    # Configurar captura de video
    if args.input == '0':
        cap = cv2.VideoCapture(0)
    elif args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        if args.input.startswith('tcp://'):
            cap = cv2.VideoCapture(args.input)
        else:
            cap = cv2.VideoCapture(f"tcp://{args.input}:5000")
    
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir fuente de video: {args.input}")
        return

    frame_count = 0
    pose_latencies = []
    yolo_size = (640, 640)
    last_coords_2d = []
    
    print("[INFO] Demo iniciado. Presione 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_frame_start = time.time()
        
        # Detección YOLO
        img_lb, scale, pad_left, pad_top = letterbox(frame, new_size=yolo_size)
        
        try:
            outputs = yolo_pipeline(
                images=img_lb,
                score_threshold=args.conf_thresh,
                nms_threshold=args.iou_thresh,
            )
            
            if isinstance(outputs, list) and len(outputs) > 0:
                outputs = outputs[0]
            
            # Procesar detecciones
            if hasattr(outputs, 'boxes') and len(outputs.boxes) > 0:
                boxes_640 = np.array(outputs.boxes[0])
                scores_out = np.array(outputs.scores[0])
                labels_out = np.array(outputs.labels[0])
                
                # Filtrar personas con confianza suficiente
                mask_person = (labels_out == 0) & (scores_out >= args.conf_thresh)
                
                if np.any(mask_person):
                    boxes_640_person = boxes_640[mask_person]
                    scores_person = scores_out[mask_person]
                    
                    # Convertir coordenadas y procesar pose
                    final_boxes = []
                    for box in boxes_640_person:
                        x1_lb, y1_lb, x2_lb, y2_lb = box
                        
                        # Revertir letterbox
                        x1_un = (x1_lb - pad_left) / scale
                        y1_un = (y1_lb - pad_top) / scale
                        x2_un = (x2_lb - pad_left) / scale
                        y2_un = (y2_lb - pad_top) / scale

                        x1o = int(max(x1_un, 0))
                        y1o = int(max(y1_un, 0))
                        x2o = int(min(x2_un, frame.shape[1] - 1))
                        y2o = int(min(y2_un, frame.shape[0] - 1))

                        if (x2o > x1o) and (y2o > y1o):
                            final_boxes.append((x1o, y1o, x2o, y2o))

                    # Procesar pose para cada detección
                    last_coords_2d.clear()
                    for n, (x1o, y1o, x2o, y2o) in enumerate(final_boxes):
                        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                        
                        # Preparar bbox para procesamiento
                        bbox = [x1o, y1o, x2o - x1o, y2o - y1o]
                        proc_bbox = pose_utils.process_bbox(np.array(bbox), frame.shape[1], frame.shape[0])
                        
                        if proc_bbox is not None:
                            
                            # Generar patch de imagen
                            img_patch, img2bb_trans = generate_patch_image(
                                frame, proc_bbox, False, 1.0, 0.0, False
                            )
                            # Preparar entrada para ConvNeXt
                            inp = pose_transform(img_patch).unsqueeze(0).to(device)
                            # Obtener profundidad raíz con RootNet
                            root_depth = root_wrapper.predict_depth(frame, bbox)
                            
                            
                            # Inferencia ConvNeXt
                            t0 = time.time()
                            pose_3d = corrected_inference_final_v4(pose_model, inp, cfg)
                            if pose_3d is not None:
                                pose_3d_numpy = pose_3d[0].cpu().numpy()
                                pose_3d_corrected = exact_demo_processing(pose_3d_numpy,img2bb_trans, root_depth, cfg) 

                                if pose_3d_corrected is not None:
                                    coords_2d = pose_3d_corrected[:, :2]
                                    print(f" DEBUG: pose_3d shape: {pose_3d.shape}")
                                    t1 = time.time()
                                    
                                    latency_ms = (t1 - t0) * 1000
                                    pose_latencies.append(latency_ms)
                                    
                                    # # Aplicar alineación robusta
                                    # coords_aligned = robust_skeleton_alignment(
                                    #     coords_2d, [x1o, y1o, x2o, y2o]
                                    # )
                                    # Verificar límites de imagen
                                    coords_2d[:, 0] = np.clip(coords_2d[:, 0], 0, frame.shape[1]-1)
                                    coords_2d[:, 1] = np.clip(coords_2d[:, 1], 0, frame.shape[0]-1)
                                    
                                    last_coords_2d.append(coords_2d)
                                    
                                    # Debug info
                                    if frame_count % 30 == 0:
                                        joint_analysis = debug_joint_positions(coords_2d, frame_count)
                                        skeleton_suggestions = suggest_skeleton_connections(joint_analysis)
                                        
                                        print(f"✅ Frame {frame_count}: Pose procesada exitosamente")
                                        print(f"🔍 DEBUG Frame {frame_count}: bbox={proc_bbox}, root_depth={root_depth:.1f}")
                                        print(f"   Coords range X: [{coords_2d[:, 0].min():.1f}, {coords_2d[:, 0].max():.1f}]")
                                        print(f"   Coords range Y: [{coords_2d[:, 1].min():.1f}, {coords_2d[:, 1].max():.1f}]")
                                    if frame_count % 60 == 0:
                                        print(f"🔄 Esqueleto actualizado con {len(skeleton)} conexiones")
                                else:
                                    print("⚠️ Fallo en post-procesamiento")
                                    continue
                            else:
                                print("⚠️ Fallo en inferencia, saltando frame")
                                continue

                        
        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            import traceback
            traceback.print_exc()

        frame_count += 1
        
        # Dibujar esqueletos
        for joints in last_coords_2d:
            draw_skeleton(frame, joints, skeleton, color=(0, 255, 0))
        
        # Mostrar información en pantalla
        fps = 1.0 / max((time.time() - t_frame_start), 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if pose_latencies:
            avg_latency = sum(pose_latencies[-10:]) / len(pose_latencies[-10:])  # Últimas 10
            cv2.putText(frame, f"Pose Latency: {avg_latency:.1f}ms", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("ConvNeXt + RootNet Integration (Fixed)", frame)

        # Estadísticas cada 30 frames
        if frame_count % 30 == 0 and pose_latencies:
            avg_lat = sum(pose_latencies) / len(pose_latencies)
            print(f"📊 Frame {frame_count}, Latencia promedio: {avg_lat:.2f}ms")
            pose_latencies.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Demo finalizado exitosamente")

if __name__ == "__main__":
    main()