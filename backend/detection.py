# -*- coding: utf-8 -*-
"""
detection.py
Détection d'objets, visages et défauts techniques
"""

import math
import numpy as np
import cv2
from config import Config, model_cache

# Imports optionnels
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_face = mp.solutions.face_detection
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ========================= YOLO =========================
def load_yolo_model():
    """Chargement lazy de YOLOv8"""
    if not YOLO_AVAILABLE:
        return None
    
    if model_cache.get('yolo') is None:
        try:
            print("📦 Chargement YOLOv8n...")
            model_cache.set('yolo', YOLO("yolov8n.pt"))
            print("✅ YOLO prêt")
        except Exception as e:
            print(f"❌ Erreur YOLO: {e}")
            return None
    
    return model_cache.get('yolo')


def detect_objects(img_rgb):
    """Détection d'objets avec YOLO"""
    model = load_yolo_model()
    
    if model is None:
        return []
    
    try:
        results = model.predict(img_rgb, imgsz=Config.YOLO_IMG_SIZE, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            
            detections.append({
                "class": model.names.get(cls, str(cls)),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })
        
        return detections
    
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return []


# ========================= MEDIAPIPE FACES =========================
def detect_faces_mediapipe(img_rgb):
    """Détection de visages avec MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    try:
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45) as face_detection:
            h, w = img_rgb.shape[:2]
            results = face_detection.process(img_rgb)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    
                    score = float(detection.score[0]) if detection.score else None
                    
                    faces.append({
                        "bbox": [x, y, bw, bh],
                        "score": score
                    })
            
            return faces
    
    except Exception as e:
        print(f"❌ MediaPipe face detection error: {e}")
        return []


# ========================= DÉFAUTS TECHNIQUES =========================
def detect_motion_blur(img):
    """Détection flou de mouvement via FFT"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    center_region = magnitude[cy-h//8:cy+h//8, cx-w//8:cx+w//8]
    
    center_energy = np.mean(center_region)
    total_energy = np.mean(magnitude)
    
    if total_energy == 0:
        return False, 0.0
    
    ratio = center_energy / total_energy
    motion_score = 1.0 - ratio
    is_motion_blur = motion_score > 0.15
    
    return bool(is_motion_blur), float(motion_score)


def detect_vignette_advanced(img):
    """Détection vignetage radial"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / r.max()
    
    inner_mask = r_norm < 0.4
    outer_mask = r_norm > 0.85
    
    if not inner_mask.any() or not outer_mask.any():
        return False, 0.0
    
    inner_mean = gray[inner_mask].mean()
    outer_mean = gray[outer_mask].mean()
    
    if inner_mean == 0:
        return False, 0.0
    
    score = (inner_mean - outer_mean) / inner_mean
    is_vignette = score > Config.VIGNETTE_THRESHOLD
    
    return bool(is_vignette), float(score)


def detect_chromatic_aberration(img):
    """Détection aberration chromatique"""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(bgr)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    if edges.sum() == 0:
        return False, 0.0
    
    edge_pixels = edges > 0
    diff_rg = np.abs(r.astype(int) - g.astype(int))[edge_pixels]
    diff_rb = np.abs(r.astype(int) - b.astype(int))[edge_pixels]
    
    if diff_rg.size == 0:
        return False, 0.0
    
    score = float((diff_rg.mean() + diff_rb.mean()) / 2.0)
    mean_intensity = float(np.mean(gray[edge_pixels]))
    
    if mean_intensity == 0:
        mean_intensity = 1.0
    
    normalized_score = score / mean_intensity
    is_aberration = normalized_score > Config.CHROMATIC_AB_THRESHOLD
    
    return bool(is_aberration), float(normalized_score)


def detect_horizon_angle(img_gray):
    """Détection angle horizon via Hough"""
    edges = cv2.Canny(img_gray, 50, 150)
    
    min_line_length = int(min(img_gray.shape) / 4)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=100, 
                            minLineLength=min_line_length, 
                            maxLineGap=20)
    
    if lines is None:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    if not angles:
        return 0.0
    
    horizontal_angles = [a for a in angles if abs(a) < 45]
    
    if not horizontal_angles:
        return 0.0
    
    return float(np.median(horizontal_angles))
