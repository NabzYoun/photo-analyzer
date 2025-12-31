"""
app.py - Backend Flask pour connecter l'interface web à ton analyzer
Installation : pip install flask flask-cors python-dotenv
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from io import BytesIO
import json
import os
import sys

# ⭐ AJOUTER CES 3 LIGNES ⭐
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ajouter le dossier backend au path Python
sys.path.insert(0, BASE_DIR)

# Import tes modules d'analyzer
from analyzer import analyze_image
from core import make_json_safe


app = Flask(__name__)
CORS(app)  # Activer CORS pour React

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Endpoint pour analyser une photo
    Reçoit : image en base64
    Retourne : résultats d'analyse en JSON
    """
    try:
        # Récupérer l'image du request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Pas d\'image fournie'}), 400
        
        # Décoder l'image base64
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = image_data.replace('data:image/webp;base64,', '')
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        # Lancer l'analyse (appelle ton analyzer.py)
        print(f"🔍 Analyse d'une image {img_rgb.shape}...")
        analysis = analyze_image_from_array(img_rgb)
        
        # Formater la réponse pour le frontend
        response = {
            'success': True,
            'sharpness': analysis.get('sharpness', 0),
            'brightness': analysis.get('brightness', 0),
            'contrast': analysis.get('contrast', 0),
            'noise': analysis.get('noise', 0),
            'quality_score': analysis.get('quality_score', 0),
            
            # Sujets détectés
            'subjects': analysis.get('subjects', []),
            
            # Style affinities (Top 5)
            'style_affinities': analysis.get('style_affinities', {}),
            
            # Composition rules avec recommandations
            'composition_rules': analysis.get('composition_rules', {}),
            'composition_score': analysis.get('composition', {}).get('composition_score', 0),
            
            # Style recommandé
            'best_style': extract_best_style(analysis),
            
            # Prompt IA généré
            'ai_prompt': generate_ai_prompt(analysis),
            
            # Conseils d'experts
            'advice': extract_advice(analysis),
            
            # Analyse complète (debug)
            'full_analysis': make_json_safe(analysis)
        }
        response = json.loads(json.dumps(response, default=str))
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

#nha 20251230
def generate_smart_caption(analysis):
    """
    Génère une description intelligente basée sur l'analyse
    si BLIP ne fonctionne pas
    """
    try:
        subjects = analysis.get('subjects', [])
        scene = analysis.get('scene', {})
        scene_type = scene.get('scene_type', 'unknown')
        brightness = analysis.get('brightness', 128)
        faces = analysis.get('faces', [])
        
        # Construction de la description
        parts = []
        
        # 1. Type de scène
        scene_descriptions = {
            'mountain': 'Une photo de montagne',
            'forest': 'Une photo en forêt',
            'beach': 'Une photo de plage',
            'urban': 'Une photo urbaine',
            'street': 'Une photo de rue',
            'indoor': 'Une photo en intérieur',
            'sky': 'Une photo avec un ciel visible',
            'nature': 'Une photo de nature',
            'landscape': 'Un paysage'
        }
        
        scene_found = False
        for key, desc in scene_descriptions.items():
            if key in scene_type.lower():
                parts.append(desc)
                scene_found = True
                break
        
        if not scene_found:
            parts.append('Une photographie')
        
        # 2. Sujets détectés
        if len(faces) > 0:
            if len(faces) == 1:
                parts.append('avec un portrait')
            else:
                parts.append(f'avec {len(faces)} personnes')
        elif subjects:
            subject_classes = [s['class'] for s in subjects[:3]]
            if 'person' in subject_classes:
                parts.append('montrant une ou plusieurs personnes')
            else:
                # Traduire les classes d'objets
                translations = {
                    'car': 'une voiture',
                    'dog': 'un chien',
                    'cat': 'un chat',
                    'tree': 'des arbres',
                    'building': 'des bâtiments',
                    'bird': 'un oiseau',
                    'flower': 'des fleurs',
                    'bicycle': 'un vélo',
                    'motorcycle': 'une moto',
                    'boat': 'un bateau'
                }
                
                translated = [translations.get(s, s) for s in subject_classes]
                if len(translated) == 1:
                    parts.append(f'avec {translated[0]}')
                elif len(translated) == 2:
                    parts.append(f'avec {translated[0]} et {translated[1]}')
                else:
                    parts.append(f'avec {", ".join(translated[:2])} et plus')
        
        # 3. Ambiance lumineuse
        if brightness < 80:
            parts.append('dans une ambiance sombre')
        elif brightness > 180:
            parts.append('très lumineuse')
        else:
            parts.append('bien éclairée')
        
        # 4. Qualité
        quality = analysis.get('quality_score', 50)
        if quality > 75:
            parts.append('de haute qualité')
        elif quality > 50:
            parts.append('de bonne qualité')
        
        # Assembler
        caption = ' '.join(parts) + '.'
        
        # Première lettre en majuscule
        caption = caption[0].upper() + caption[1:]
        
        return caption
        
    except Exception as e:
        print(f"Erreur generate_smart_caption: {e}")
        return "Photographie professionnelle analysée."


def analyze_image_from_array(img_rgb):
    """
    Version simplifiée d'analyze_image() qui prend un array numpy
    au lieu d'un chemin fichier
    """
    import cv2
    from core import (
        compute_brightness, compute_contrast, compute_sharpness,
        estimate_noise_luminance, compute_saturation
    )
    from detection import (
        detect_objects, detect_faces_mediapipe,
        detect_motion_blur, detect_vignette_advanced,
        detect_chromatic_aberration, detect_horizon_angle
    )
    from analysis import (
        dominant_colors, analyze_composition, build_zone_report
    )
    from ai_models import predict_scene, compute_all_style_affinities, compute_quality_score, blip_caption
    from composition_rules import CompositionAnalyzer
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Métriques basiques
    brightness = compute_brightness(img_rgb)
    contrast = compute_contrast(img_rgb)
    sharpness = compute_sharpness(img_rgb)
    noise = estimate_noise_luminance(img_rgb)
    saturation = compute_saturation(img_rgb)
    
    # Détections
    is_blurry = sharpness < 80.0
    motion_blur_detected, motion_blur_score = detect_motion_blur(img_rgb)
    vignette_flag, vign_score = detect_vignette_advanced(img_rgb)
    chrom_ab_flag, chrom_score = detect_chromatic_aberration(img_rgb)
    horizon_angle = detect_horizon_angle(gray)
    
    # Couleurs et objets
    colors = dominant_colors(img_rgb)
    subjects = detect_objects(img_rgb)
    faces = detect_faces_mediapipe(img_rgb)
    
    # Composition et scène
    comp = analyze_composition(img_rgb, subjects)
    scene = predict_scene(img_rgb)
    
    # 🆕 Génération du caption (légende)


    
    # Zones
    try:
        zones = build_zone_report(img_rgb, subjects=subjects, scene_type=scene.get('scene_type', 'unknown'))
    except Exception as e:
        print(f"Erreur zones: {e}")
        zones = []
    
    # Construction du dict analysis
    analysis = {
        'path': 'web_upload',
        'width': int(img_rgb.shape[1]),
        'height': int(img_rgb.shape[0]),
        'brightness': round(float(brightness), 2),
        'contrast': round(float(contrast), 2),
        'sharpness': round(float(sharpness), 2),
        'is_blurry': bool(is_blurry),
        'motion_blur_detected': bool(motion_blur_detected),
        'noise': round(float(noise), 2),
        'dominant_colors': colors,
        'vignette': bool(vignette_flag),
        'chrom_ab': (bool(chrom_ab_flag), round(float(chrom_score), 3)),
        'subjects': subjects,
        'faces': faces,
        'composition': comp,
        'scene': scene,
       # 'caption': caption,  # 🆕 AJOUT
        'horizon_angle': round(float(horizon_angle), 2),
        'zones': zones
    }
    # ✅ NOUVELLE SECTION - Génération caption
caption = None

# Essayer BLIP d'abord
try:
    print("  💬 Tentative BLIP...")
    caption, error = blip_caption(img_rgb)
    print(f"  BLIP résultat: caption={caption}, error={error}")
    
    if error or not caption or caption.strip() == "" or caption == "Image d'analyse":
        print("  ⚠️ BLIP invalide, passage au fallback")
        caption = None
except Exception as e:
    print(f"  ❌ Erreur BLIP: {e}")
    caption = None

# Fallback intelligent si BLIP a échoué
if not caption:
    print("  🤖 Génération caption intelligent...")
    caption = generate_smart_caption(analysis)
    print(f"  ✅ Caption final: {caption}")

# Ajouter le caption au dict
    analysis['caption'] = caption
    
    # Score de qualité
    quality = compute_quality_score(analysis)
    analysis['quality_score'] = quality
    
    # Style affinities (Top 5)
    style_affinities = compute_all_style_affinities(img_rgb, analysis)
    analysis['style_affinities'] = style_affinities
    
    # Analyse composition rules
    try:
        composition_analyzer = CompositionAnalyzer()
        composition_rules_analysis = composition_analyzer.analyze_all_rules(img_rgb, subjects, analysis)
        analysis['composition_rules'] = composition_rules_analysis
    except Exception as e:
        print(f"Erreur composition rules: {e}")
        analysis['composition_rules'] = {}
    
    return analysis

def extract_best_style(analysis):
    """Extraire le meilleur style pour l'affichage"""
    try:
        style_affinities = analysis.get('style_affinities', {})
        best_match = style_affinities.get('best_match')
        
        if best_match:
            return {
                'label': best_match.get('label', 'Unknown'),
                'description': best_match.get('description', ''),
                'score': round(best_match.get('score', 0), 3),
                'category': best_match.get('category', ''),
                'difficulty': best_match.get('difficulty', '')
            }
    except:
        pass
    
    return None


#nha
def generate_caption_fallback(analysis):
    subjects = analysis.get("subjects", [])
    scene = analysis.get("scene", {}).get("scene_type", "")
    brightness = analysis.get("brightness", 128)
    
    parts = []

    if subjects:
        main = subjects[0].get("class", "sujet")
        parts.append(f"Photographie mettant en scène un {main}")

    if scene:
        parts.append(f"dans un contexte {scene.replace('_', ' ')}")

    if brightness < 80:
        parts.append("avec une ambiance sombre")
    elif brightness > 180:
        parts.append("avec une ambiance lumineuse")
    else:
        parts.append("avec une lumière équilibrée")

    return " ".join(parts).capitalize() + "."
#finnha


def generate_ai_prompt(analysis):
    """
    Générer un prompt pour Midjourney/DALL-E basé sur l'analyse
    C'est LA fonction clé de ton différenciation !
    """
    try:
        # Récupérer infos
        scene = analysis.get('scene', {})
        scene_type = scene.get('scene_type', 'unknown').lower()
        brightness = analysis.get('brightness', 128)
        contrast = analysis.get('contrast', 50)
        subjects = analysis.get('subjects', [])
        faces = analysis.get('faces', [])
        best_style = extract_best_style(analysis)
        
        # Construction du prompt
        prompt_parts = []
        
        # 1. Type de photo
        if len(faces) > 0:
            if len(faces) == 1:
                prompt_parts.append("portrait")
            else:
                prompt_parts.append("group portrait")
        elif any(s.get('class') == 'person' for s in subjects):
            prompt_parts.append("photograph of person")
        else:
            prompt_parts.append("photograph")
        
        # 2. Style si disponible
        if best_style:
            style_label = best_style['label'].lower()
            prompt_parts.append(f"in the style of {style_label}")
        
        # 3. Lighting
        if brightness < 80:
            prompt_parts.append("moody lighting, dark atmosphere")
        elif brightness > 180:
            prompt_parts.append("bright, well-lit, golden hour lighting")
        else:
            prompt_parts.append("soft natural lighting")
        
        # 4. Scene/location
        scene_mapping = {
            'mountain': 'mountain landscape',
            'forest': 'forest environment',
            'beach': 'beach setting',
            'urban': 'urban street',
            'street': 'city street',
            'indoor': 'indoor studio',
            'sky': 'outdoor with sky'
        }
        
        for key, val in scene_mapping.items():
            if key in scene_type:
                prompt_parts.append(val)
                break
        
        # 5. Qualité
        sharpness = analysis.get('sharpness', 100)
        if sharpness > 150:
            prompt_parts.append("sharp, detailed, professional")
        else:
            prompt_parts.append("artistic, soft focus")
        
        # 6. Composition
        comp_score = analysis.get('composition', {}).get('composition_score', 50)
        if comp_score > 70:
            prompt_parts.append("rule of thirds composition")
        
        # Joindre
        prompt = ", ".join(prompt_parts)
        prompt = prompt[0].upper() + prompt[1:] if prompt else "Professional photograph"
        prompt += ". 8k, professional photography"
        
        return prompt
    
    except Exception as e:
        print(f"Erreur prompt: {e}")
        return "Professional photograph, natural lighting, sharp details, 8k"


def extract_advice(analysis):
    """Extraire les conseils"""
    advice = []
    
    brightness = analysis.get('brightness', 128)
    contrast = analysis.get('contrast', 50)
    sharpness = analysis.get('sharpness', 100)
    noise = analysis.get('noise', 20)
    
    # Ajouter les conseils pertinents
    if sharpness < 100:
        advice.append("🔍 Augmente la netteté avec Clarity dans Lightroom")
    
    if brightness < 80:
        advice.append("☀️ La photo est sombre, augmente l'exposition")
    elif brightness > 180:
        advice.append("⚡ La photo est surexposée, réduis les highlights")
    
    if contrast < 30:
        advice.append("📊 Augmente le contraste pour plus d'impact")
    
    if noise > 40:
        advice.append("🔇 Bruit élevé, utilise la réduction de bruit")
    
    if analysis.get('vignette'):
        advice.append("🌓 Vignetage détecté, active la correction du profil")
    
    if analysis.get('motion_blur_detected'):
        advice.append("⚠️ Flou de mouvement détecté, utilise une vitesse plus rapide")
    
    if not advice:
        advice.append("✨ Excellente photo technique !")
    
    return advice[:5]  # Max 5 conseils


# Routes de test
@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'API running', 'version': '1.0'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )