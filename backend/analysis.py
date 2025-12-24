# -*- coding: utf-8 -*-
"""
analysis.py
Analyse de composition, couleurs et segmentation en zones (VERSION AMÉLIORÉE V2)
"""

import numpy as np
import cv2
from config import Config

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ========================= COULEURS =========================
def dominant_colors(img, n_colors=None):
    """Extraction couleurs dominantes"""
    if n_colors is None:
        n_colors = Config.N_DOMINANT_COLORS
    
    small = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    
    if SKLEARN_AVAILABLE:
        kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
        kmeans.fit(data)
        
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        items = sorted(zip(counts, centers), reverse=True)
        
        return [
            {"count": int(c), "color": col.tolist()}
            for c, col in items
        ]
    else:
        data_int = (data / 32).astype(int)
        keys = (data_int[:, 0] * 64 * 64 + 
                data_int[:, 1] * 64 + 
                data_int[:, 2])
        
        unique, counts = np.unique(keys, return_counts=True)
        idx = counts.argsort()[::-1][:n_colors]
        
        result = []
        for i in idx:
            key = unique[i]
            b = (key // (64 * 64)) * 32
            g = ((key // 64) % 64) * 32
            r = (key % 64) * 32
            result.append({"count": int(counts[i]), "color": [int(r), int(g), int(b)]})
        
        return result


# ========================= COMPOSITION =========================
def analyze_composition(img_rgb, subjects):
    """Analyse composition règle des tiers"""
    h, w = img_rgb.shape[:2]
    
    thirds = {
        "vertical_left": w / 3,
        "vertical_right": 2 * w / 3,
        "horizontal_top": h / 3,
        "horizontal_bottom": 2 * h / 3,
    }
    
    if not subjects:
        return {
            "thirds": thirds,
            "subject_position": None,
            "composition_score": 50.0
        }
    
    main_subject = max(subjects, 
                      key=lambda s: (s["bbox"][2] - s["bbox"][0]) * 
                                   (s["bbox"][3] - s["bbox"][1]))
    
    x1, y1, x2, y2 = main_subject["bbox"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    tolerance = 20
    position = {
        "center": [float(cx), float(cy)],
        "on_left_third": cx < thirds["vertical_left"] + tolerance,
        "on_right_third": cx > thirds["vertical_right"] - tolerance,
        "on_top_third": cy < thirds["horizontal_top"] + tolerance,
        "on_bottom_third": cy > thirds["horizontal_bottom"] - tolerance,
    }
    
    score = 60
    
    if position["on_left_third"] or position["on_right_third"]:
        score += 20
    if position["on_top_third"] or position["on_bottom_third"]:
        score += 20
    
    if len(subjects) > 1:
        score += 5
    
    score = min(score, 100)
    
    return {
        "thirds": thirds,
        "subject_position": position,
        "composition_score": float(score)
    }


# ========================= SEGMENTATION =========================
def segment_image_simple(img_rgb, small_size=None, n_clusters=None):
    """Segmentation légère KMeans"""
    if small_size is None:
        small_size = Config.SEGMENT_SIZE
    if n_clusters is None:
        n_clusters = Config.N_CLUSTERS
    
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (small_size, small_size), interpolation=cv2.INTER_AREA)
    Z = small.reshape((-1, 3)).astype(np.float32)
    
    if not SKLEARN_AVAILABLE:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return [{"zone_id": 0, "mask": mask, "percent_area": 100.0}]
    
    k = min(n_clusters, 6)
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
    labels = kmeans.fit_predict(Z)
    
    labels_img = labels.reshape((small_size, small_size))
    labels_img = cv2.resize(labels_img.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    zones = []
    for cid in range(k):
        mask = (labels_img == cid).astype(np.uint8) * 255
        area = float(np.sum(mask > 0)) / (w * h) * 100.0
        zones.append({
            "zone_id": int(cid),
            "mask": mask,
            "percent_area": round(area, 2)
        })
    
    return zones


def classify_zone_intelligent(img_rgb, mask, subjects, scene_type, zone_metrics):
    """
    Classification intelligente de zone basée sur le contexte
    VERSION AMÉLIORÉE avec meilleure détection
    
    Args:
        img_rgb: Image RGB
        mask: Masque de la zone
        subjects: Objets détectés par YOLO
        scene_type: Type de scène (Places365)
        zone_metrics: Métriques de la zone (brightness, saturation, position)
    
    Returns:
        label: "subject", "foreground", "background", "sky", "ground", etc.
    """
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    
    if len(ys) == 0:
        return "unknown"
    
    # Position verticale moyenne (0 = haut, 1 = bas)
    y_mean = float(np.mean(ys)) / float(h)
    x_mean = float(np.mean(xs)) / float(w)
    
    # Métriques de la zone
    brightness = zone_metrics.get('brightness', 0)
    saturation = zone_metrics.get('saturation', 0)
    area_percent = zone_metrics.get('area_percent', 0)
    sharpness = zone_metrics.get('sharpness', 0)
    
    # 🔥 DÉTECTION DU SUJET PRINCIPAL
    contains_subject = False
    subject_class = None
    subject_confidence = 0
    
    # Calcul de la surface totale de la zone
    zone_area = np.sum(mask > 0)
    
    for subj in subjects:
        x1, y1, x2, y2 = subj['bbox']
        conf = subj.get('confidence', 0)
        
        # Seuil de confiance minimal (YOLO peut halluciner)
        if conf < 0.5:
            continue
        
        # Calculer le chevauchement entre le bbox et la zone
        subject_mask = np.zeros((h, w), dtype=np.uint8)
        x1_safe = int(max(0, min(w-1, x1)))
        y1_safe = int(max(0, min(h-1, y1)))
        x2_safe = int(max(0, min(w, x2)))
        y2_safe = int(max(0, min(h, y2)))
        
        subject_mask[y1_safe:y2_safe, x1_safe:x2_safe] = 255
        
        overlap = np.sum((mask > 0) & (subject_mask > 0))
        subject_area = np.sum(subject_mask > 0)
        
        # Besoin d'OVERLAP significatif ET confiance élevée
        if subject_area > 0 and overlap > 0:
            overlap_ratio = overlap / subject_area
            
            # Pour être classé comme "person", besoin de 50% de chevauchement ET confiance > 0.6
            if subject_class == 'person' and overlap_ratio > 0.5 and conf > 0.6:
                contains_subject = True
                subject_confidence = conf
            # Pour autres objets, critères moins stricts
            elif subject_class != 'person' and overlap_ratio > 0.4 and conf > 0.5:
                if conf > subject_confidence:
                    contains_subject = True
                    subject_class = subj.get('class', 'object')
                    subject_confidence = conf
    
    if contains_subject:
        # Zone contenant le sujet principal détecté
        if subject_class in ['cat', 'dog', 'bird', 'horse']:
            return f"pet_{subject_class}"
        elif subject_class == 'person':
            return "person"
        else:
            return "main_subject"
    
    # 🔥 DÉTECTION HEURISTIQUE : Si zone grosse + sombre + au centre = probablement le sujet
    # (cas où YOLO rate la détection)
    if area_percent > 40 and brightness < 100 and abs(x_mean - 0.5) < 0.25:
        # Grosse zone sombre, pas loin du centre = probablement le sujet (silhouette, etc.)
        if saturation > 50:
            return "dark_subject_colored"
        else:
            return "dark_subject"
    
    # 🔥 DÉTECTION BASÉE SUR LE TYPE DE SCÈNE
    scene_lower = scene_type.lower() if scene_type else ''
    
    # Scènes d'intérieur
    indoor_scenes = ['bedroom', 'livingroom', 'kitchen', 'office', 'bathroom', 'room', 'studio']
    is_indoor = any(indoor in scene_lower for indoor in indoor_scenes)
    
    if is_indoor:
        # En intérieur, pas de "sky", seulement mur/sol/meuble
        if y_mean < 0.35:
            if brightness > 160:
                return "ceiling_bright"
            elif brightness > 100:
                return "wall_bright"
            else:
                return "wall_dark"
        elif y_mean > 0.70:
            if brightness < 80:
                return "floor_dark"
            elif brightness < 120:
                return "floor"
            else:
                return "floor_bright"
        else:
            # Zone centrale
            if saturation > 80:
                return "furniture_colored"
            elif area_percent > 30:
                return "furniture_large"
            else:
                return "furniture"
    
    # 🔥 SCÈNES D'EXTÉRIEUR
    outdoor_scenes = ['mountain', 'field', 'forest', 'beach', 'sky', 'ocean', 'nature', 'park']
    is_outdoor = any(outdoor in scene_lower for outdoor in outdoor_scenes)
    
    if is_outdoor:
        if y_mean < 0.35 and brightness > 140:
            return "sky"
        elif y_mean > 0.65:
            if 'beach' in scene_lower or 'sand' in scene_lower:
                return "sand"
            elif 'water' in scene_lower or 'ocean' in scene_lower:
                return "water"
            else:
                return "ground"
        else:
            if saturation > 70:
                return "vegetation"
            else:
                return "middle_ground"
    
    # 🔥 CLASSIFICATION BASÉE SUR LA LUMINOSITÉ (Si pas d'indice de scène)
    # MEILLEURE DÉTECTION : zones sombres vs zones claires
    
    if brightness < 40:
        # Très sombre = probablement ombre ou zone sombre intentionnelle
        if sharpness > 150:
            return "shadow_with_detail"
        else:
            return "deep_shadow"
    
    elif brightness < 80:
        # Sombre mais avec détails
        if area_percent > 40:
            return "dark_subject"  # Sujet sombre (personne en silhouette, etc.)
        else:
            return "dark_area"
    
    elif brightness < 120:
        # Zone moyennement sombre
        if saturation > 60:
            return "colored_midtone"
        else:
            return "midtone"
    
    elif brightness < 180:
        # Zone claire normal
        if saturation > 80 and area_percent > 20:
            return "bright_colored"
        elif sharpness > 200:
            return "bright_sharp"
        else:
            return "bright_area"
    
    else:
        # Très claire
        if saturation > 70:
            return "highlight_colored"
        else:
            return "highlight"


def analyze_zone(img_rgb, mask, label):
    """Métriques par zone"""
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    pix = masked[mask > 0]
    
    if pix.size == 0:
        return {}
    
    hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
    hsv_pix = hsv[mask > 0]
    
    brightness = float(np.mean(pix))
    saturation = float(np.mean(hsv_pix[:, 1])) if hsv_pix.size else 0.0
    contrast = float(np.std(pix))
    
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    percent_area = float(np.sum(mask > 0)) / (img_rgb.shape[0] * img_rgb.shape[1]) * 100.0
    
    return {
        "label": label,
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "percent_area": round(percent_area, 2)
    }


def zone_advice_intelligent(zone):
    """Conseils retouche intelligents par zone - VERSION AMÉLIORÉE"""
    advice = []
    label = zone.get("label", "zone")
    bri = zone.get("brightness", 0)
    sha = zone.get("sharpness", 0)
    sat = zone.get("saturation", 0)
    area = zone.get("percent_area", 0)
    
    # DEBUG: Si label="person" mais brightness très basse, c'est une erreur
    if label == "person" and bri < 40:
        # Redétecter comme dark_subject
        label = "dark_subject"
    
    # 🔥 CONSEILS SPÉCIFIQUES PAR TYPE DE ZONE (AMÉLIORÉ)
    
    # ====== SUJETS ======
    if label.startswith("pet_"):
        # Zone contenant un animal
        if sha < 150:
            advice.append("Augmenter netteté sur l'animal (+20 clarity)")
        if bri < 100:
            advice.append("Éclaircir l'animal légèrement (+0.3 EV)")
        advice.append("S'assurer que les yeux sont nets et ont un catchlight")
    
    elif label == "main_subject":
        if sha < 100:
            advice.append("Augmenter netteté du sujet (+15 sharpness)")
        if bri < 90:
            advice.append("Augmenter exposition du sujet (+0.5 EV)")
    
    elif label == "person":
        if sat > 100:
            advice.append("Réduire saturation peau (-5 à -10)")
        if sha > 200:
            advice.append("Adoucir légèrement la peau (-5 clarity)")
    
    elif label == "dark_subject":
        advice.append("Sujet sombre : augmenter les shadows (+20)")
        advice.append("Ou utiliser fill-flash en post-prod (dodge local)")
    
    elif label == "dark_subject_colored":
        advice.append("Sujet sombre coloré : augmenter les shadows (+20)")
        advice.append("Considérer boost de saturation (vibrance +10)")
        advice.append("Éviter sur-saturer les tons foncés")
    
    elif label.startswith("shadow"):
        if "detail" in label:
            advice.append("Excellentes ombres avec détails : RAS")
        else:
            advice.append("Ombres pures : récupérer détails (+15 shadows)")
    
    # ====== CIEL / EXTÉRIEUR ======
    elif label == "sky":
        if bri > 170:
            advice.append("Réduire highlights ciel (-30 à -50)")
        if sha < 20:
            advice.append("Ajouter Dehaze (+10 à +20)")
    
    elif label == "vegetation":
        advice.append("Paysage naturel : augmenter vibrance (+10)")
        advice.append("Considérer boost des verts (HSL)")
    
    elif label == "water":
        advice.append("Eau : réduire highlights pour détails")
        advice.append("Augmenter saturation blues/cyans (HSL)")
    
    elif label == "sand":
        advice.append("Sable : chauds généralement OK")
        advice.append("Ajouter texture avec Clarity si plat")
    
    elif label == "ground":
        if sha > 100:
            advice.append("Texture présente : OK")
        else:
            advice.append("Sol peu texturé : ajouter Dehaze")
    
    # ====== INTÉRIEUR ======
    elif label.startswith("wall"):
        if bri > 200:
            advice.append("Mur très clair : réduire exposition (-0.3 EV local)")
        advice.append("Considérer assombrir le fond pour isoler le sujet")
    
    elif label.startswith("ceiling"):
        advice.append("Plafond : généralement dans les ombres OK")
        if bri > 180:
            advice.append("Plafond trop clair : réduire pour équilibre")
    
    elif label.startswith("floor"):
        if sha > 100:
            advice.append("Sol très texturé : OK")
        else:
            advice.append("Sol peu texturé : peut flouter légèrement")
    
    elif label.startswith("furniture"):
        if "colored" in label:
            advice.append("Meubles colorés : maintenir saturation")
        else:
            advice.append("Arrière-plan neutre : RAS")
    
    # ====== ZONES CLAIRES / SOMBRES ======
    elif label.startswith("bright"):
        if "colored" in label:
            advice.append("Zone claire colorée : vérifier saturation")
        else:
            advice.append("Vérifier qu'il n'y a pas d'écrêtage")
    
    elif label.startswith("highlight"):
        advice.append("Très claire : réduire highlights (-20 à -30)")
        advice.append("Vérifier présence de détails dans les blancs")
    
    elif label == "dark_area":
        if area > 30:
            advice.append("Large zone sombre : voir si récupérable (+20 shadows)")
        else:
            advice.append("Zone sombre ponctuelle : probablement intentionnelle")
    
    elif label.startswith("midtone"):
        advice.append("Zone neutre : RAS généralement")
    
    else:
        advice.append("Évaluer en contexte de l'image globale")
    
    return advice if advice else ["Évaluer en contexte"]


def build_zone_report(img_rgb, subjects=None, scene_type="unknown"):
    """Assemblage rapport zones avec contexte intelligent"""
    if subjects is None:
        subjects = []
    
    zones_raw = segment_image_simple(img_rgb)
    final = []
    
    for z in zones_raw:
        mask = z.get("mask")
        
        # Calculer métriques temporaires pour la classification
        temp_metrics = analyze_zone(img_rgb, mask, "temp")
        temp_metrics['area_percent'] = z.get("percent_area")
        
        # Classification intelligente
        label = classify_zone_intelligent(
            img_rgb, 
            mask, 
            subjects, 
            scene_type, 
            temp_metrics
        )
        
        # Métriques finales
        metrics = analyze_zone(img_rgb, mask, label)
        
        if metrics:
            metrics["zone_id"] = z.get("zone_id")
            metrics["percent_area"] = z.get("percent_area")
            metrics["advice"] = zone_advice_intelligent(metrics)
            final.append(metrics)
    
    return final