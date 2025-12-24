# -*- coding: utf-8 -*-
"""
core.py
Métriques techniques et gestion des images
"""

import os
import json
import math
import numpy as np
import cv2
from PIL import Image, ImageOps
from fractions import Fraction

try:
    from PIL.TiffImagePlugin import IFDRational
except ImportError:
    IFDRational = None


# ========================= UTILITAIRES =========================
def make_json_safe(obj):
    """Conversion robuste pour JSON"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if IFDRational and isinstance(obj, IFDRational):
        return float(obj)
    if isinstance(obj, Fraction):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def download_file(url, dest, show=True):
    """Téléchargement avec progression"""
    import urllib.request
    try:
        if show:
            print(f"📥 Téléchargement : {os.path.basename(url)}...")
        urllib.request.urlretrieve(url, dest)
        if show:
            print(f"✅ Sauvegardé : {dest}")
        return True
    except Exception as e:
        if show:
            print(f"❌ Erreur : {e}")
        return False


# ========================= IMAGE I/O =========================
def load_image_rgb(path, max_size=1400, auto_rotate=True):
    """Chargement optimisé avec gestion EXIF"""
    with Image.open(path) as pil:
        if auto_rotate:
            try:
                pil = ImageOps.exif_transpose(pil)
            except Exception:
                pass
        img = np.array(pil.convert("RGB"))
    
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def extract_exif(path):
    """Extraction EXIF avec gestion GPS"""
    try:
        with Image.open(path) as pil:
            raw = pil._getexif()
            if not raw:
                return {}
            
            from PIL.ExifTags import TAGS
            exif_data = {TAGS.get(k, k): v for k, v in raw.items()}
        
        result = {}
        
        result["Make"] = exif_data.get("Make")
        result["Model"] = exif_data.get("Model")
        result["DateTime"] = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
        
        fn = exif_data.get("FNumber")
        if fn:
            try:
                result["FNumber"] = round(fn[0] / fn[1], 2) if isinstance(fn, tuple) else float(fn)
            except (TypeError, ZeroDivisionError):
                result["FNumber"] = str(fn)
        
        et = exif_data.get("ExposureTime")
        if et:
            result["ExposureTime"] = f"{et[0]}/{et[1]}" if isinstance(et, tuple) else str(et)
        
        iso = exif_data.get("ISOSpeedRatings") or exif_data.get("PhotographicSensitivity")
        if iso:
            try:
                result["ISO"] = int(iso[0]) if isinstance(iso, (list, tuple)) else int(iso)
            except (TypeError, ValueError):
                result["ISO"] = str(iso)
        
        fl = exif_data.get("FocalLength")
        if fl:
            try:
                result["FocalLength"] = round(fl[0] / fl[1], 1) if isinstance(fl, tuple) else float(fl)
            except (TypeError, ZeroDivisionError):
                result["FocalLength"] = str(fl)
        
        gps = exif_data.get("GPSInfo")
        if gps:
            try:
                def to_degrees(values):
                    d = values[0][0] / values[0][1]
                    m = values[1][0] / values[1][1]
                    s = values[2][0] / values[2][1]
                    return d + m / 60.0 + s / 3600.0
                
                if 2 in gps and 4 in gps:
                    lat = to_degrees(gps[2])
                    lon = to_degrees(gps[4])
                    
                    if gps.get(1, 'N') == 'S':
                        lat = -lat
                    if gps.get(3, 'E') == 'W':
                        lon = -lon
                    
                    result["GPS"] = {"lat": round(lat, 6), "lon": round(lon, 6)}
            except (KeyError, TypeError, ZeroDivisionError):
                pass
        
        return result
    except Exception as e:
        return {}


# ========================= MÉTRIQUES TECHNIQUES =========================
def compute_brightness(img):
    """Luminosité moyenne en LAB"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return float(np.mean(lab[:, :, 0]))


def compute_contrast(img):
    """Contraste via écart-type"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def compute_sharpness(img, roi=None):
    """Netteté via Laplacien"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if roi:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_saturation(img):
    """Saturation moyenne HSV"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 1]))


def compute_dynamic_range(img):
    """Plage dynamique P5-P95"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    
    p5 = np.searchsorted(cumsum, 0.05 * total)
    p95 = np.searchsorted(cumsum, 0.95 * total)
    
    return float(p95 - p5)


def compute_softness(img):
    """Douceur (inverse netteté)"""
    sharp = compute_sharpness(img)
    return 1.0 / (1.0 + sharp / 100.0)


def compute_texture_score(img):
    """Score de texture via Sobel"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(magnitude))


def compute_depth_score(img):
    """Estimation profondeur via zones"""
    h, w = img.shape[:2]
    zone_height = h // 3
    
    zones_sharpness = []
    for i in range(3):
        y_start = i * zone_height
        y_end = (i + 1) * zone_height if i < 2 else h
        zone = img[y_start:y_end, :]
        zones_sharpness.append(compute_sharpness(zone))
    
    return float(np.std(zones_sharpness))


def estimate_noise_luminance(img):
    """Estimation bruit par patchs"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    patch_size = 64
    
    stds = []
    for y in range(0, max(1, h - patch_size + 1), patch_size):
        for x in range(0, max(1, w - patch_size + 1), patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            if patch.size > 0:
                stds.append(float(np.std(patch)))
    
    return float(np.median(stds)) if stds else float(np.std(gray))


def is_black_and_white(img, threshold=10):
    """Détection N&B"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:, :, 1]) < threshold


def compute_color_harmony(img):
    """Harmonie via entropie teintes"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = hist.flatten() / hist.sum()
    
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    normalized_entropy = entropy / 7.49
    
    return 1.0 - normalized_entropy


def compute_composition_balance(img, subjects):
    """Équilibre composition"""
    h, w = img.shape[:2]
    
    if not subjects:
        return 0.5
    
    weights = []
    positions = []
    
    for subj in subjects:
        x1, y1, x2, y2 = subj["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        
        positions.append([cx / w, cy / h])
        weights.append(area)
    
    if not weights:
        return 0.5
    
    positions = np.array(positions)
    weights = np.array(weights) / np.sum(weights)
    
    center_of_mass = np.sum(positions * weights[:, np.newaxis], axis=0)
    distance_to_center = np.sqrt(np.sum((center_of_mass - 0.5)**2))
    
    thirds_positions = [1/3, 2/3]
    min_dist_third = min([abs(center_of_mass[0] - t) for t in thirds_positions] + 
                         [abs(center_of_mass[1] - t) for t in thirds_positions])
    
    balance_score = 1.0 - min(distance_to_center, min_dist_third)
    return float(np.clip(balance_score, 0, 1))


def compute_composition_simplicity(img):
    """Simplicité composition"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return float(1.0 - min(edge_density * 10, 1.0))


def compute_geometry_score(img):
    """Score géométrie (lignes)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=100, 
                            minLineLength=int(min(img.shape[:2]) / 8),
                            maxLineGap=20)
    
    if lines is None:
        return 0.0
    
    num_lines = len(lines)
    normalized = num_lines / (img.shape[0] * img.shape[1] / 10000)
    return float(min(normalized, 1.0))


def normalize_score(value, min_val, max_val, clamp=True):
    """Normalisation 0-1"""
    if max_val == min_val:
        return 0.5
    
    score = (value - min_val) / (max_val - min_val)
    
    if clamp:
        score = max(0.0, min(1.0, score))
    
    return round(float(score), 3)
