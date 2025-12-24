# -*- coding: utf-8 -*-
"""
ai_models.py
Modèles IA : BLIP, Places365, Style Matcher avec système de contraintes enrichi
"""

import os
import json
import numpy as np
from pathlib import Path
from config import Config, model_cache
from core import download_file, normalize_score

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

from PIL import Image


# ========================= PLACES365 =========================
def load_places365_classes(path=None):
    """Chargement classes Places365"""
    if path is None:
        path = Config.PLACES_CLASSES_FILE
    
    if not os.path.exists(path):
        print(f"⚠️ {path} introuvable, téléchargement...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
        if download_file(url, path, show=True):
            print(f"✅ Fichier téléchargé")
        else:
            return None
    
    classes = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_name = parts[0].split("/")[-1]
                    classes.append(class_name)
        
        print(f"✅ {len(classes)} classes Places365 chargées")
        return classes
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


def predict_scene(img_rgb):
    """Prédiction scène avec Places365"""
    if not TORCH_AVAILABLE:
        return {"scene_type": "unknown", "probability": 0.0, "top_5": []}
    
    classes = model_cache.get('places_classes')
    if classes is None:
        classes = load_places365_classes()
        if classes is None:
            return {"scene_type": "unknown", "probability": 0.0, "top_5": []}
        model_cache.set('places_classes', classes)
    
    model = model_cache.get('places_model')
    if model is None:
        try:
            print("📦 Chargement Places365...")
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            
            model_path = "resnet50_places365.pth.tar"
            if not os.path.exists(model_path):
                if not download_file(Config.PLACES_MODEL_URL, model_path):
                    return {"scene_type": "unknown", "probability": 0.0, "top_5": []}
            
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.eval()
            
            model_cache.set('places_model', model)
            print("✅ Places365 prêt")
        except Exception as e:
            print(f"❌ Erreur Places365: {e}")
            return {"scene_type": "unknown", "probability": 0.0, "top_5": []}
    
    try:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        pil = Image.fromarray(img_rgb)
        input_tensor = transform(pil).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        top5_prob, top5_idx = torch.topk(probs, 5)
        
        return {
            "scene_type": classes[top5_idx[0]],
            "probability": float(top5_prob[0]),
            "top_5": [
                {"scene": classes[idx], "prob": float(prob)}
                for idx, prob in zip(top5_idx.tolist(), top5_prob.tolist())
            ]
        }
    except Exception as e:
        print(f"❌ Prédiction failed: {e}")
        return {"scene_type": "unknown", "probability": 0.0, "top_5": []}


# ========================= BLIP =========================
def blip_caption(img_rgb, max_tokens=64):
    """Génération légende avec BLIP"""
    if not BLIP_AVAILABLE:
        return None, "BLIP not available"
    
    processor = model_cache.get('blip_processor')
    model = model_cache.get('blip_model')
    
    if model is None:
        try:
            print("📦 Chargement BLIP...")
            model_id = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id)
            model.eval()
            
            model_cache.set('blip_processor', processor)
            model_cache.set('blip_model', model)
            print("✅ BLIP prêt")
        except Exception as e:
            print(f"❌ Erreur BLIP: {e}")
            return None, str(e)
    
    try:
        pil = Image.fromarray(img_rgb.astype("uint8"), "RGB")
        inputs = processor(images=pil, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
        return caption, None
    except Exception as e:
        return None, str(e)


# ========================= STYLE MATCHER AVEC CONTRAINTES ENRICHIES =========================
def load_style_profiles(path=None):
    """Chargement profils stylistiques"""
    if path is None:
        path = Config.STYLES_PROFILES_FILE
    
    print(f"🔍 Tentative de chargement : {path}")
    print(f"🔍 Chemin absolu : {os.path.abspath(path)}")
    
    profiles = model_cache.get('style_profiles')
    if profiles is not None:
        print(f"✅ Profils déjà en cache : {len(profiles)}")
        return profiles
    
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return []
    
    try:
        print(f"📖 Lecture du fichier...")
        with open(path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        if not isinstance(profiles, list):
            print(f"❌ Format invalide : attendu liste, reçu {type(profiles)}")
            return []
        
        model_cache.set('style_profiles', profiles)
        print(f"✅ {len(profiles)} profils stylistiques chargés")
        
        if profiles:
            print(f"   Premier profil : {profiles[0].get('label', 'N/A')}")
        
        return profiles
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON : {e}")
        return []
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return []


def check_scene_constraints(scene_type, constraints):
    """Vérifie les contraintes liées à la scène"""
    scene_lower = scene_type.lower()
    
    # Paysages
    landscape_scenes = ['mountain', 'field', 'forest', 'lake', 'ocean', 'valley', 
                       'canyon', 'desert', 'beach', 'waterfall', 'sky', 'coast',
                       'glacier', 'snowfield', 'cliff', 'hill', 'volcano', 'ridge']
    is_landscape = any(ls in scene_lower for ls in landscape_scenes)
    
    # 🆕 DÉTECTION AMÉLIORÉE : Si la scène contient "sky" et pas "indoor" = paysage probable
    if 'sky' in scene_lower and 'indoor' not in scene_lower and 'building' not in scene_lower:
        is_landscape = True
    
    # Urbain
    urban_scenes = ['street', 'building', 'city', 'downtown', 'alley', 'plaza', 
                   'square', 'crosswalk', 'sidewalk', 'urban']
    is_urban = any(us in scene_lower for us in urban_scenes)
    
    # Extérieur
    outdoor_scenes = landscape_scenes + ['park', 'garden', 'courtyard', 'highway']
    is_outdoor = any(os in scene_lower for os in outdoor_scenes)
    
    # 🆕 Si pas d'indices d'intérieur et ciel visible = extérieur
    indoor_keywords = ['room', 'indoor', 'hall', 'corridor', 'kitchen', 'office']
    has_indoor = any(kw in scene_lower for kw in indoor_keywords)
    if 'sky' in scene_lower and not has_indoor:
        is_outdoor = True
    
    # Architecture
    architecture_scenes = ['building', 'facade', 'skyscraper', 'bridge', 'tower', 
                          'monument', 'church', 'cathedral']
    is_architecture = any(ar in scene_lower for ar in architecture_scenes)
    
    # Intérieur
    indoor_scenes = ['bedroom', 'livingroom', 'kitchen', 'office', 'restaurant', 
                    'bar', 'library', 'museum', 'gallery', 'studio']
    is_indoor = any(ind in scene_lower for ind in indoor_scenes)
    
    checks = []
    
    if constraints.get('requires_landscape'):
        checks.append(('landscape', is_landscape, "Doit être un paysage"))
    
    if constraints.get('requires_urban'):
        checks.append(('urban', is_urban, "Doit être urbain"))
    
    if constraints.get('requires_outdoor'):
        checks.append(('outdoor', is_outdoor, "Doit être en extérieur"))
    
    if constraints.get('requires_architecture'):
        checks.append(('architecture', is_architecture, "Doit contenir de l'architecture"))
    
    if constraints.get('requires_indoor_or_studio'):
        checks.append(('indoor', is_indoor, "Doit être en intérieur/studio"))
    
    return checks


def compute_advanced_metrics(img_rgb, analysis):
    """Métriques avancées pour style matching"""
    from core import (compute_saturation, compute_dynamic_range, compute_softness,
                     compute_texture_score, compute_depth_score, is_black_and_white,
                     compute_color_harmony, compute_composition_balance,
                     compute_composition_simplicity, compute_geometry_score)
    
    metrics = {}
    
    metrics['brightness'] = analysis.get('brightness', 0)
    metrics['contrast'] = analysis.get('contrast', 0)
    metrics['sharpness'] = analysis.get('sharpness', 0)
    metrics['noise'] = analysis.get('noise', 0)
    
    print("    🔬 Métriques avancées...")
    metrics['saturation'] = compute_saturation(img_rgb)
    metrics['dynamic_range'] = compute_dynamic_range(img_rgb)
    metrics['softness'] = compute_softness(img_rgb)
    metrics['texture_score'] = compute_texture_score(img_rgb)
    metrics['depth_score'] = compute_depth_score(img_rgb)
    metrics['black_and_white'] = is_black_and_white(img_rgb)
    
    if not metrics['black_and_white']:
        metrics['color_harmony'] = compute_color_harmony(img_rgb)
        metrics['color_saturation'] = normalize_score(metrics['saturation'], *Config.SATURATION_RANGE)
        metrics['vibrance'] = normalize_score(metrics['saturation'] * 1.2, *Config.SATURATION_RANGE)
        metrics['color_complexity'] = 1.0 - metrics['color_harmony']
        metrics['color_desaturation'] = 1.0 - normalize_score(metrics['saturation'], *Config.SATURATION_RANGE)
    else:
        metrics['color_harmony'] = 0.0
        metrics['color_saturation'] = 0.0
        metrics['vibrance'] = 0.0
        metrics['color_complexity'] = 0.0
        metrics['color_desaturation'] = 1.0
    
    subjects = analysis.get('subjects', [])
    metrics['composition_balance'] = compute_composition_balance(img_rgb, subjects)
    metrics['composition_simplicity'] = compute_composition_simplicity(img_rgb)
    metrics['geometry_score'] = compute_geometry_score(img_rgb)
    
    # Métriques dérivées
    metrics['moment_score'] = 0.5
    metrics['skin_tone_score'] = 0.5
    metrics['light_directionality'] = normalize_score(metrics['contrast'], 20, 80)
    metrics['shadow_detail'] = normalize_score(255 - metrics['brightness'], 50, 150)
    metrics['local_contrast'] = metrics['contrast']
    metrics['visual_noise'] = metrics['texture_score']
    metrics['subject_isolation'] = 1.0 - metrics['composition_balance']
    metrics['color_shift'] = 0.1
    
    # Métriques de scène
    metrics['scene_type'] = analysis.get('scene', {}).get('scene_type', 'unknown')
    metrics['num_faces'] = len(analysis.get('faces', []))
    metrics['num_objects'] = len(subjects)
    
    # Détection présence humaine (visages + personnes détectées)
    metrics['has_human'] = metrics['num_faces'] > 0 or any(
        s.get('class', '').lower() == 'person' for s in subjects
    )
    
    # 🆕 DÉTECTION ANIMAUX DOMESTIQUES
    pet_classes = ['cat', 'dog', 'bird', 'horse']
    metrics['has_pet'] = any(
        s.get('class', '').lower() in pet_classes for s in subjects
    )
    metrics['pet_type'] = next(
        (s.get('class', '') for s in subjects if s.get('class', '').lower() in pet_classes),
        None
    )
    
    print(f"   Has pet : {metrics['has_pet']} (type: {metrics['pet_type']})")
    
    print(f"\n📊 Valeurs des métriques clés :")
    print(f"   Saturation : {metrics['saturation']:.1f}")
    print(f"   Brightness : {metrics['brightness']:.1f}")
    print(f"   Contrast : {metrics['contrast']:.1f}")
    print(f"   Sharpness : {metrics['sharpness']:.1f}")
    print(f"   Scene : {metrics['scene_type']}")
    print(f"   Faces : {metrics['num_faces']}")
    print(f"   Objects : {metrics['num_objects']}")
    print(f"   Human presence : {metrics['has_human']}")
    
    return metrics


def calculate_style_affinity(metrics, profile):
    """Calcul affinité avec un profil incluant contraintes enrichies"""
    profile_id = profile.get('id', 'unknown')
    profile_label = profile.get('label', 'Unknown')
    
    weights = profile.get('weights', {})
    
    if not weights:
        print(f"   ⚠️  {profile_label}: Pas de weights")
        return 0.0
    
    constraints = profile.get('constraints', {})
    
    print(f"\n🔍 {profile_label}:")
    if constraints:
        print(f"   Contraintes : {list(constraints.keys())}")
    
    # ============= VÉRIFICATION DES CONTRAINTES STRICTES =============
    
    # Contrainte N&B vs Couleur
    if constraints.get('black_and_white') and not metrics.get('black_and_white'):
        print(f"   ❌ Éliminé : Image en couleur mais profil N&B requis")
        return 0.0
    if constraints.get('color') and metrics.get('black_and_white'):
        print(f"   ❌ Éliminé : Image N&B mais profil couleur requis")
        return 0.0
    
    # Contrainte visage requis
    if constraints.get('requires_face'):
        num_faces = metrics.get('num_faces', 0)
        min_faces = constraints.get('min_faces', 1)
        print(f"   🔍 Visages détectés : {num_faces} (min requis : {min_faces})")
        if num_faces < min_faces:
            print(f"   ❌ Éliminé : Pas assez de visages")
            return 0.0
    
    # Contrainte aucun visage
    if constraints.get('requires_no_face'):
        num_faces = metrics.get('num_faces', 0)
        print(f"   🔍 Visages détectés : {num_faces} (aucun autorisé)")
        if num_faces > 0:
            print(f"   ❌ Éliminé : Visage(s) présent(s) mais interdit(s)")
            return 0.0
    
    # Contrainte aucun humain
    if constraints.get('requires_no_human'):
        has_human = metrics.get('has_human', False)
        print(f"   🔍 Présence humaine : {has_human} (interdite)")
        if has_human:
            print(f"   ❌ Éliminé : Présence humaine détectée mais interdite")
            return 0.0
    
    # Contrainte présence humaine requise
    if constraints.get('requires_human_presence'):
        has_human = metrics.get('has_human', False)
        print(f"   🔍 Présence humaine : {has_human} (requise)")
        if not has_human:
            print(f"   ❌ Éliminé : Aucune présence humaine détectée")
            return 0.0
    
    # 🆕 CONTRAINTE ANIMAL DOMESTIQUE REQUIS
    if constraints.get('requires_pet'):
        has_pet = metrics.get('has_pet', False)
        pet_type = metrics.get('pet_type', 'unknown')
        print(f"   🔍 Animal domestique : {has_pet} (type: {pet_type})")
        if not has_pet:
            print(f"   ❌ Éliminé : Aucun animal domestique détecté")
            return 0.0
    
    # 🆕 CONTRAINTE AUCUN ANIMAL (pour produits, food, etc.)
    if constraints.get('requires_no_pet'):
        has_pet = metrics.get('has_pet', False)
        if has_pet:
            print(f"   ❌ Éliminé : Animal présent mais interdit (produit/food/cinematic)")
            return 0.0
    
    # Note : allows_pet=true signifie que les animaux sont acceptés mais pas requis
    # (pas besoin de vérification, c'est juste permissif)
    
    # Contraintes de scène
    scene_checks = check_scene_constraints(metrics.get('scene_type', ''), constraints)
    for check_name, passed, message in scene_checks:
        print(f"   🔍 {check_name.capitalize()} : {passed}")
        if not passed:
            print(f"   ❌ Éliminé : {message}")
            return 0.0
    
    # 🆕 CONTRAINTE AUCUN PAYSAGE (pour macro, abstrait, etc.)
    if constraints.get('requires_no_landscape'):
        is_landscape = any(
            check[0] == 'landscape' and check[1] 
            for check in scene_checks
        )
        if is_landscape:
            print(f"   ❌ Éliminé : Paysage détecté mais interdit (style macro/abstrait)")
            return 0.0
    
    # Contrainte nombre d'objets maximum
    if constraints.get('max_objects'):
        num_objects = metrics.get('num_objects', 0)
        max_objects = constraints['max_objects']
        print(f"   🔍 Objets détectés : {num_objects} (max : {max_objects})")
        if num_objects > max_objects:
            print(f"   ❌ Éliminé : Trop d'objets ({num_objects} > {max_objects})")
            return 0.0
    
    # Contrainte nombre d'objets minimum
    if constraints.get('min_objects'):
        num_objects = metrics.get('num_objects', 0)
        min_objects = constraints['min_objects']
        print(f"   🔍 Objets détectés : {num_objects} (min : {min_objects})")
        if num_objects < min_objects:
            print(f"   ❌ Éliminé : Pas assez d'objets ({num_objects} < {min_objects})")
            return 0.0
    
    # Contraintes de valeurs minimales/maximales
    value_constraints = [
        ('min_contrast', 'contrast', 'Contraste trop faible'),
        ('max_contrast', 'contrast', 'Contraste trop élevé'),
        ('min_brightness', 'brightness', 'Trop sombre'),
        ('max_brightness', 'brightness', 'Trop lumineux'),
        ('min_saturation', 'saturation', 'Pas assez saturé'),
        ('max_saturation', 'saturation', 'Trop saturé'),
        ('min_sharpness', 'sharpness', 'Pas assez net'),
        ('max_sharpness', 'sharpness', 'Trop net'),
        ('min_texture', 'texture_score', 'Pas assez de texture'),
    ]
    
    for constraint_key, metric_key, message in value_constraints:
        if constraint_key in constraints:
            threshold = constraints[constraint_key]
            value = metrics.get(metric_key, 0)
            
            if constraint_key.startswith('min_'):
                if value < threshold:
                    print(f"   ❌ Éliminé : {message} ({value:.1f} < {threshold})")
                    return 0.0
            else:  # max_
                if value > threshold:
                    print(f"   ❌ Éliminé : {message} ({value:.1f} > {threshold})")
                    return 0.0
    
    # ============= CALCUL DU SCORE SI TOUTES LES CONTRAINTES PASSENT =============
    
    print(f"   ✅ Toutes les contraintes passées, calcul du score...")
    
    score = 0.0
    total_weight = 0.0
    missing_metrics = []
    used_metrics = []
    
    for metric_name, weight in weights.items():
        metric_value = metrics.get(metric_name)
        
        if metric_value is None:
            missing_metrics.append(metric_name)
            continue
        
        used_metrics.append((metric_name, metric_value, weight))
        
        # Normalisation des métriques brutes
        if metric_name in ['brightness', 'contrast', 'sharpness', 'noise', 'saturation', 'dynamic_range']:
            if metric_name == 'brightness':
                normalized = normalize_score(metric_value, *Config.BRIGHTNESS_RANGE)
            elif metric_name == 'contrast':
                normalized = normalize_score(metric_value, *Config.CONTRAST_RANGE)
            elif metric_name == 'sharpness':
                normalized = normalize_score(metric_value, *Config.SHARPNESS_RANGE)
            elif metric_name == 'noise':
                normalized = 1.0 - normalize_score(metric_value, *Config.NOISE_RANGE)
            elif metric_name == 'saturation':
                normalized = normalize_score(metric_value, *Config.SATURATION_RANGE)
            elif metric_name == 'dynamic_range':
                normalized = normalize_score(metric_value, *Config.DYNAMIC_RANGE_RANGE)
            else:
                normalized = metric_value
        else:
            # Pour les métriques déjà normalisées ou hors plage
            if metric_value > 1.0:
                if 'saturation' in metric_name or 'vibrance' in metric_name:
                    normalized = normalize_score(metric_value, *Config.SATURATION_RANGE)
                else:
                    normalized = min(1.0, max(0.0, metric_value))
            else:
                normalized = metric_value
        
        # Application du poids (positif ou négatif)
        if weight > 0:
            score += normalized * weight
            total_weight += abs(weight)
        else:
            score += (1.0 - normalized) * abs(weight)
            total_weight += abs(weight)
    
    # Debug pour certains profils
    if profile_id in ['ansel_adams', 'cartier_bresson', 'dramatic_portrait']:
        print(f"   📊 Métriques utilisées : {len(used_metrics)}/{len(weights)}")
        if missing_metrics:
            print(f"   ⚠️  Métriques manquantes : {', '.join(missing_metrics)}")
        print(f"   Score brut : {score:.3f}")
        print(f"   Total poids : {total_weight:.3f}")
    
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.0
    
    print(f"   🎯 Score final : {final_score:.3f} ({int(final_score*100)}%)")
    
    return round(float(np.clip(final_score, 0, 1)), 3)


def compute_all_style_affinities(img_rgb, analysis):
    """Calcul affinité avec tous les profils"""
    profiles = load_style_profiles()
    
    if not profiles:
        return {}
    
    metrics = compute_advanced_metrics(img_rgb, analysis)
    
    affinities = []
    
    print(f"\n🎨 Évaluation de {len(profiles)} profils stylistiques...")
    print("="*60)
    
    for profile in profiles:
        score = calculate_style_affinity(metrics, profile)
        
        affinity_data = {
            "id": profile.get("id"),
            "label": profile.get("label"),
            "category": profile.get("category"),
            "difficulty": profile.get("difficulty"),
            "score": score,
            "description": profile.get("description"),
            "key_characteristics": profile.get("key_characteristics", []),
            "recommended_subjects": profile.get("recommended_subjects", []),
            "editing_guidelines": profile.get("editing_guidelines", []),
            "lightroom_preset": profile.get("lightroom_preset", {}),
            "target_values": profile.get("target_values", {}),
            "pro_tips": profile.get("pro_tips", []),
            "common_mistakes": profile.get("common_mistakes", [])
        }
        
        affinities.append(affinity_data)
    
    affinities.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 RÉSULTATS FINAUX")
    print("="*60)
    for i, aff in enumerate(affinities[:5], 1):
        print(f"{i}. {aff['label']:35s} {int(aff['score']*100):3d}%")
    print("="*60)
    
    return {
        "all_styles": affinities,
        "top_5": affinities[:5],
        "best_match": affinities[0] if affinities else None,
        "metrics_used": {k: round(v, 3) if isinstance(v, float) else v 
                        for k, v in metrics.items()}
    }


def compute_quality_score(analysis):
    """Score qualité global"""
    score = 50
    
    sharp = analysis.get("sharpness", 0)
    if sharp > 150:
        score += 25
    elif sharp > 80:
        score += 10
    else:
        score -= 10
    
    bright = analysis.get("brightness", 120)
    if 90 < bright < 160:
        score += 10
    elif bright < 60 or bright > 200:
        score -= 10
    
    noise = analysis.get("noise", 20)
    if noise < 20:
        score += 10
    elif noise > 40:
        score -= 15
    
    comp_score = analysis.get("composition", {}).get("composition_score", 50)
    score += (comp_score - 50) * 0.3
    
    if analysis.get("vignette"):
        score -= 5
    
    chrom = analysis.get("chrom_ab")
    if isinstance(chrom, (list, tuple)) and chrom[0]:
        score -= 5
    
    if analysis.get("motion_blur_detected"):
        score -= 10
    
    return int(max(0, min(100, score)))