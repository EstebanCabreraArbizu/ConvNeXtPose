#!/usr/bin/env python3
"""
test_screen_capture.py - Test simple de captura de pantalla
"""

import cv2
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mss
    logger.info("✅ MSS disponible")
except ImportError:
    logger.error("❌ MSS no disponible")
    exit(1)

def test_screen_capture():
    """Test básico de captura de pantalla"""
    logger.info("🚀 Iniciando test de captura de pantalla...")
    
    # Inicializar MSS
    sct = mss.mss()
    monitor = sct.monitors[1]  # Monitor principal
    
    logger.info(f"📺 Monitor: {monitor['width']}x{monitor['height']}")
    
    # Capturar algunos frames
    for i in range(5):
        logger.info(f"📸 Capturando frame {i+1}/5...")
        
        # Capturar
        start_time = time.time()
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        capture_time = (time.time() - start_time) * 1000
        
        logger.info(f"   Frame shape: {frame.shape}")
        logger.info(f"   Capture time: {capture_time:.1f}ms")
        
        # Mostrar frame
        small_frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Screen Capture Test', small_frame)
        
        # Esperar
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    logger.info("✅ Test completado")

if __name__ == "__main__":
    test_screen_capture()
