# -*- coding: utf-8 -*-
"""
improved_captioning.py
Système de légendage amélioré avec plusieurs modèles et validation
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


def generate_smart_caption(img_rgb, analysis: Dict, subjects: List[Dict], faces: List[Dict]) -> Dict:
    """
    Génère une légende intelligente basée sur l'analyse complète
    Au lieu de se fier aveuglément à BLIP
    
    Returns:
        {
            'caption': str,
            'confidence': float,
            'source': str,
            'elements_detected': list
        }
    """
    
    elements = []
    confidence_factors = []
    
    # ========================= DÉTECTION VISAGES =========================
    num_faces = len(faces)
    if num_faces > 0:
        if num_faces == 1:
            elements.append("portrait")
            confidence_factors.append(0.9)
        else:
            elements.append(f"photo de groupe ({num_faces} personnes)")
            confidence_factors.append(0.85)
    
    # ========================= DÉTECTION OBJETS YOLO =========================
    detected_objects = {}
    for subj in subjects:
        obj_class = subj.get('class', '').lower()
        confidence = subj.get('confidence', 0)
        
        if confidence > 0.5:  # Seuil de confiance
            if obj_class not in detected_objects or detected_objects[obj_class] < confidence:
                detected_objects[obj_class] = confidence
    
    # Traduction des objets en français
    object_translations = {
        'person': 'personne',
        'cat': 'chat',
        'dog': 'chien',
        'bird': 'oiseau',
        'car': 'voiture',
        'bicycle': 'vélo',
        'bottle': 'bouteille',
        'cup': 'tasse',
        'book': 'livre',
        'laptop': 'ordinateur portable',
        'phone': 'téléphone',
        'chair': 'chaise',
        'couch': 'canapé',
        'bed': 'lit',
        'table': 'table'
    }
    
    for obj, conf in sorted(detected_objects.items(), key=lambda x: x[1], reverse=True)[:3]:
        obj_fr = object_translations.get(obj, obj)
        elements.append(obj_fr)
        confidence_factors.append(conf)
    
    # ========================= ANALYSE SCÈNE =========================
    scene = analysis.get('scene', {})
    scene_type = scene.get('scene_type', '')
    scene_prob = scene.get('probability', 0)
    
    if scene_prob > 0.4:
        scene_translations = {
            'bedroom': 'chambre',
            'livingroom': 'salon',
            'kitchen': 'cuisine',
            'bathroom': 'salle de bain',
            'office': 'bureau',
            'street': 'rue',
            'mountain': 'montagne',
            'forest': 'forêt',
            'beach': 'plage',
            'ocean': 'océan',
            'sky': 'ciel'
        }
        
        scene_fr = scene_translations.get(scene_type.lower(), scene_type)
        elements.append(f"scène de {scene_fr}")
        confidence_factors.append(scene_prob)
    
    # ========================= ANALYSE COULEURS =========================
    brightness = analysis.get('brightness', 128)
    is_bw = analysis.get('black_and_white', False)
    
    if is_bw or brightness < 80:
        elements.append("ambiance sombre")
        confidence_factors.append(0.9 if brightness < 60 else 0.7)
    elif brightness > 180:
        elements.append("lumière vive")
        confidence_factors.append(0.8)
    
    # ========================= CONSTRUCTION LÉGENDE =========================
    if not elements:
        caption = "Photo"
        confidence = 0.3
        source = "generic"
    else:
        # Priorité aux éléments les plus confiants
        main_elements = elements[:3]
        
        if num_faces > 0:
            # Portrait prioritaire
            if num_faces == 1:
                caption = f"Portrait en {main_elements[1] if len(main_elements) > 1 else 'intérieur'}"
            else:
                caption = f"Photo de groupe de {num_faces} personnes"
        else:
            # Pas de visage, décrire les objets/scène
            caption = " avec ".join(main_elements[:2]).capitalize()
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        source = "smart_analysis"
    
    return {
        'caption': caption,
        'confidence': round(confidence, 2),
        'source': source,
        'elements_detected': elements
    }


def validate_blip_caption(blip_caption: str, analysis: Dict, subjects: List[Dict], faces: List[Dict]) -> Dict:
    """
    Valide et corrige une légende BLIP basée sur l'analyse réelle
    
    Returns:
        {
            'original': str,
            'validated': str,
            'corrections': list,
            'confidence': float,
            'is_reliable': bool
        }
    """
    
    corrections = []
    confidence = 1.0
    
    if not blip_caption:
        return {
            'original': '',
            'validated': '',
            'corrections': ['Aucune légende générée'],
            'confidence': 0.0,
            'is_reliable': False
        }
    
    caption_lower = blip_caption.lower()
    
    # ========================= VÉRIFICATION VISAGES =========================
    num_faces = len(faces)
    
    # Détecte les mentions de personnes dans BLIP
    person_keywords = ['man', 'woman', 'person', 'people', 'boy', 'girl', 'child', 'kid']
    mentions_person = any(kw in caption_lower for kw in person_keywords)
    
    if mentions_person and num_faces == 0:
        corrections.append(f"⚠️ BLIP mentionne une personne mais aucun visage détecté")
        confidence *= 0.3
    
    if num_faces > 0 and not mentions_person:
        corrections.append(f"✓ {num_faces} visage(s) détecté(s) mais non mentionné par BLIP")
    
    # ========================= VÉRIFICATION VÊTEMENTS =========================
    clothing_keywords = ['suit', 'tie', 'shirt', 'dress', 'jacket', 'coat']
    mentions_clothing = any(kw in caption_lower for kw in clothing_keywords)
    
    if mentions_clothing:
        corrections.append("⚠️ BLIP mentionne des vêtements (souvent imprécis)")
        confidence *= 0.6
    
    # ========================= VÉRIFICATION OBJETS =========================
    detected_objects = [s.get('class', '').lower() for s in subjects if s.get('confidence', 0) > 0.5]
    
    # Objets couramment mal détectés par BLIP
    blip_objects = ['tie', 'book', 'bottle', 'cup', 'laptop', 'phone']
    for obj in blip_objects:
        if obj in caption_lower and obj not in detected_objects:
            corrections.append(f"⚠️ BLIP mentionne '{obj}' mais non détecté par YOLO")
            confidence *= 0.7
    
    # ========================= VÉRIFICATION SCÈNE =========================
    scene_type = analysis.get('scene', {}).get('scene_type', '').lower()
    
    # Incohérences de lieu
    if 'outdoor' in caption_lower and any(indoor in scene_type for indoor in ['room', 'kitchen', 'office']):
        corrections.append("⚠️ BLIP dit 'outdoor' mais scène détectée comme intérieur")
        confidence *= 0.4
    
    # ========================= VÉRIFICATION COULEURS =========================
    brightness = analysis.get('brightness', 128)
    
    if 'bright' in caption_lower and brightness < 80:
        corrections.append("⚠️ BLIP dit 'bright' mais photo sombre")
        confidence *= 0.5
    
    # ========================= VALIDATION FINALE =========================
    is_reliable = confidence > 0.6 and len(corrections) < 3
    
    validated_caption = blip_caption
    
    if not is_reliable:
        # Générer une légende alternative
        smart_caption = generate_smart_caption(None, analysis, subjects, faces)
        validated_caption = f"{smart_caption['caption']} (légende corrigée)"
        corrections.append(f"→ Légende originale peu fiable, remplacée par analyse smart")
    
    return {
        'original': blip_caption,
        'validated': validated_caption,
        'corrections': corrections,
        'confidence': round(confidence, 2),
        'is_reliable': is_reliable
    }


def generate_multilevel_caption(img_rgb, analysis: Dict, subjects: List[Dict], faces: List[Dict]) -> Dict:
    """
    Génère plusieurs niveaux de légendes avec différents niveaux de détail
    
    Returns:
        {
            'simple': str,      # "Portrait"
            'detailed': str,    # "Portrait d'une personne en intérieur"
            'technical': str,   # "Portrait vertical, lumière naturelle, ambiance sombre"
            'confidence': float
        }
    """
    
    smart = generate_smart_caption(img_rgb, analysis, subjects, faces)
    elements = smart['elements_detected']
    
    # Niveau 1 : Simple
    if len(faces) > 0:
        simple = "Portrait" if len(faces) == 1 else "Photo de groupe"
    elif any('chat' in e or 'chien' in e for e in elements):
        simple = "Photo animalière"
    else:
        simple = elements[0].capitalize() if elements else "Photo"
    
    # Niveau 2 : Détaillé
    detailed = smart['caption']
    
    # Niveau 3 : Technique
    technical_parts = []
    
    # Orientation
    width = analysis.get('width', 0)
    height = analysis.get('height', 0)
    if height > width * 1.2:
        technical_parts.append("format vertical")
    elif width > height * 1.2:
        technical_parts.append("format horizontal")
    else:
        technical_parts.append("format carré")
    
    # Lumière
    brightness = analysis.get('brightness', 128)
    if brightness < 80:
        technical_parts.append("ambiance sombre")
    elif brightness > 180:
        technical_parts.append("haute lumière")
    else:
        technical_parts.append("exposition équilibrée")
    
    # Type de photo
    if elements:
        technical_parts.append(elements[0])
    
    technical = ", ".join(technical_parts).capitalize()
    
    return {
        'simple': simple,
        'detailed': detailed,
        'technical': technical,
        'confidence': smart['confidence']
    }


def caption_with_context(img_rgb, analysis: Dict, subjects: List[Dict], faces: List[Dict], 
                        blip_caption: Optional[str] = None) -> Dict:
    """
    Système complet de légendage avec validation et contexte
    
    Returns:
        {
            'caption': str,              # Légende finale retenue
            'caption_simple': str,       # Version simple
            'caption_detailed': str,     # Version détaillée
            'caption_technical': str,    # Version technique
            'blip_original': str,        # Légende BLIP brute
            'validation': dict,          # Résultat validation BLIP
            'confidence': float,         # Confiance globale
            'source': str,              # Source de la légende
            'warnings': list            # Avertissements
        }
    """
    
    # Génération multi-niveaux
    multilevels = generate_multilevel_caption(img_rgb, analysis, subjects, faces)
    
    warnings = []
    
    # Validation BLIP si disponible
    if blip_caption:
        validation = validate_blip_caption(blip_caption, analysis, subjects, faces)
        
        if validation['is_reliable']:
            # BLIP est fiable
            final_caption = blip_caption
            source = "blip_validated"
            confidence = validation['confidence']
        else:
            # BLIP peu fiable, utiliser smart caption
            final_caption = multilevels['detailed']
            source = "smart_analysis"
            confidence = multilevels['confidence']
            warnings.append("Légende BLIP corrigée car peu fiable")
            warnings.extend(validation['corrections'][:2])
    else:
        # Pas de BLIP, utiliser smart caption
        validation = None
        final_caption = multilevels['detailed']
        source = "smart_analysis"
        confidence = multilevels['confidence']
    
    return {
        'caption': final_caption,
        'caption_simple': multilevels['simple'],
        'caption_detailed': multilevels['detailed'],
        'caption_technical': multilevels['technical'],
        'blip_original': blip_caption or '',
        'validation': validation,
        'confidence': confidence,
        'source': source,
        'warnings': warnings
    }


# ========================= INTÉGRATION DANS ANALYZER.PY =========================
"""
Dans analyzer.py, remplacer :

    caption, caption_src = blip_caption(img_rgb)
    if caption is None:
        caption = ""
        caption_src = "none"

Par :

    # Légende BLIP (optionnelle)
    blip_result, _ = blip_caption(img_rgb)
    
    # Système de légendes amélioré
    from improved_captioning import caption_with_context
    
    caption_complete = caption_with_context(
        img_rgb, 
        analysis, 
        subjects, 
        faces, 
        blip_caption=blip_result
    )
    
    analysis["caption_complete"] = caption_complete
    analysis["caption"] = caption_complete["caption"]
    analysis["caption_source"] = caption_complete["source"]
    
    # Affichage dans le terminal
    if caption_complete['warnings']:
        print("  ⚠️  Avertissements légende:")
        for warning in caption_complete['warnings'][:3]:
            print(f"      • {warning}")
"""