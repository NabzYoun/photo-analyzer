"""
app.py - Avec Hugging Face Inference API
Installation : pip install requests
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
from PIL import Image
import io
import requests

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analyzer import analyze_image
from core import make_json_safe

app = Flask(__name__)
CORS(app)


# ==========================================
# HUGGING FACE API
# ==========================================

# 🔑 Remplacez par votre token Hugging Face
# Obtenez-le sur : https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN = "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"

# Modèles disponibles (vous pouvez en tester d'autres)
HF_MODELS = {
    'blip2': 'Salesforce/blip2-opt-2.7b',
    'blip': 'Salesforce/blip-image-captioning-large',
    'git': 'microsoft/git-large-coco'
}


def generate_caption_huggingface(img_rgb, model_name='blip'):
    """
    Génère une description avec l'API Hugging Face Inference
    
    Args:
        img_rgb: Image numpy array RGB
        model_name: 'blip2', 'blip', ou 'git'
    """
    try:
        # Vérifier le token
        if HUGGINGFACE_API_TOKEN == "hf_VotreTokenIci":
            print("  ⚠️ Token Hugging Face non configuré")
            return None, "Token non configuré"
        
        print(f"  🤗 Génération avec Hugging Face ({model_name})...")
        
        # Convertir l'image en bytes
        img_pil = Image.fromarray(img_rgb)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        
        # URL de l'API Hugging Face
        model_id = HF_MODELS.get(model_name, HF_MODELS['blip'])
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Headers avec votre token
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
        }
        
        # Envoyer la requête
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Le format de réponse varie selon le modèle
            if isinstance(result, list) and len(result) > 0:
                caption_en = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                caption_en = result.get('generated_text', '')
            else:
                caption_en = str(result)
            
            # Nettoyer et traduire
            caption_en = caption_en.strip()
            caption_fr = translate_simple(caption_en)
            
            print(f"  ✅ Hugging Face: {caption_fr}")
            return caption_fr, None
            
        elif response.status_code == 503:
            # Modèle en cours de chargement
            print("  ⏳ Modèle en cours de chargement, réessayez dans 20s")
            return None, "Modèle en chargement"
            
        else:
            error_msg = f"Erreur API: {response.status_code}"
            print(f"  ❌ {error_msg}")
            return None, error_msg
        
    except requests.exceptions.Timeout:
        print("  ❌ Timeout de l'API")
        return None, "Timeout"
    except Exception as e:
        print(f"  ❌ Erreur Hugging Face: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def translate_simple(english_text):
    """Traduction simple anglais -> français"""
    trans = {
        # Personnes
        'a man': 'un homme',
        'a woman': 'une femme',
        'a person': 'une personne',
        'a boy': 'un garçon',
        'a girl': 'une fille',
        'people': 'des personnes',
        
        # Vêtements et accessoires
        'wearing': 'portant',
        'a hat': 'un chapeau',
        'a cap': 'une casquette',
        'glasses': 'des lunettes',
        'sunglasses': 'des lunettes de soleil',
        'a shirt': 'une chemise',
        'a jacket': 'une veste',
        'a suit': 'un costume',
        'a dress': 'une robe',
        
        # Actions
        'smiling': 'souriant',
        'standing': 'debout',
        'sitting': 'assis',
        'walking': 'marchant',
        'running': 'courant',
        'looking at': 'regardant',
        'holding': 'tenant',
        'posing': 'posant',
        
        # Lieux
        'in front of': 'devant',
        'behind': 'derrière',
        'next to': 'à côté de',
        'near': 'près de',
        'a wall': 'un mur',
        'a building': 'un bâtiment',
        'a house': 'une maison',
        'a street': 'une rue',
        'the street': 'la rue',
        'outside': 'à l\'extérieur',
        'inside': 'à l\'intérieur',
        'indoor': 'en intérieur',
        'outdoor': 'en extérieur',
        
        # Objets
        'a car': 'une voiture',
        'a tree': 'un arbre',
        'trees': 'des arbres',
        'the sky': 'le ciel',
        'a phone': 'un téléphone',
        'a camera': 'un appareil photo',
        'a table': 'une table',
        'a chair': 'une chaise',
        
        # Couleurs
        'red': 'rouge',
        'blue': 'bleu',
        'green': 'vert',
        'yellow': 'jaune',
        'black': 'noir',
        'white': 'blanc',
        'gray': 'gris',
        
        # Autres
        'with': 'avec',
        'and': 'et',
        'on': 'sur',
        'of': 'de',
        'the': 'le/la',
        'a': 'un/une'
    }
    
    # Traduire
    text_fr = english_text.lower()
    for en, fr in trans.items():
        text_fr = text_fr.replace(en, fr)
    
    # Capitaliser la première lettre
    text_fr = text_fr[0].upper() + text_fr[1:] if text_fr else english_text
    
    return text_fr


# ==========================================
# FALLBACK INTELLIGENT
# ==========================================

def generate_smart_caption(analysis):
    """Génère description basée sur l'analyse"""
    try:
        subjects = analysis.get('subjects', [])
        scene = analysis.get('scene', {})
        scene_type = scene.get('scene_type', 'unknown')
        brightness = analysis.get('brightness', 128)
        faces = analysis.get('faces', [])
        
        parts = []
        
        # Type de scène
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
        
        # Sujets
        if len(faces) > 0:
            parts.append('avec un portrait' if len(faces) == 1 else f'avec {len(faces)} personnes')
        elif subjects:
            subject_classes = [s['class'] for s in subjects[:3]]
            if 'person' in subject_classes:
                parts.append('montrant une ou plusieurs personnes')
            else:
                translations = {
                    'car': 'une voiture', 'dog': 'un chien', 'cat': 'un chat',
                    'tree': 'des arbres', 'building': 'des bâtiments', 'bird': 'un oiseau',
                    'flower': 'des fleurs', 'bicycle': 'un vélo', 'motorcycle': 'une moto'
                }
                
                translated = [translations.get(s, s) for s in subject_classes]
                if len(translated) == 1:
                    parts.append(f'avec {translated[0]}')
                elif len(translated) == 2:
                    parts.append(f'avec {translated[0]} et {translated[1]}')
                else:
                    parts.append(f'avec {", ".join(translated[:2])} et plus')
        
        # Ambiance
        if brightness < 80:
            parts.append('dans une ambiance sombre')
        elif brightness > 180:
            parts.append('très lumineuse')
        else:
            parts.append('bien éclairée')
        
        # Qualité
        quality = analysis.get('quality_score', 50)
        if quality > 75:
            parts.append('de haute qualité')
        elif quality > 50:
            parts.append('de bonne qualité')
        
        caption = ' '.join(parts) + '.'
        caption = caption[0].upper() + caption[1:]
        
        return caption
        
    except Exception as e:
        print(f"Erreur generate_smart_caption: {e}")
        return "Photographie professionnelle analysée."


# ==========================================
# ANALYSE IMAGE
# ==========================================

def analyze_image_from_array(img_rgb):
    """Analyse avec Hugging Face API + Fallback"""
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
    from ai_models import (
        predict_scene, compute_all_style_affinities,
        compute_quality_score, blip_caption
    )
    from composition_rules import CompositionAnalyzer

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Métriques
    brightness = compute_brightness(img_rgb)
    contrast = compute_contrast(img_rgb)
    sharpness = compute_sharpness(img_rgb)
    noise = estimate_noise_luminance(img_rgb)
    saturation = compute_saturation(img_rgb)

    is_blurry = sharpness < 80.0
    motion_blur_detected, motion_blur_score = detect_motion_blur(img_rgb)
    vignette_flag, vign_score = detect_vignette_advanced(img_rgb)
    chrom_ab_flag, chrom_score = detect_chromatic_aberration(img_rgb)
    horizon_angle = detect_horizon_angle(gray)

    colors = dominant_colors(img_rgb)
    subjects = detect_objects(img_rgb)
    faces = detect_faces_mediapipe(img_rgb)

    comp = analyze_composition(img_rgb, subjects)
    scene = predict_scene(img_rgb)

    try:
        zones = build_zone_report(img_rgb, subjects=subjects, scene_type=scene.get('scene_type', 'unknown'))
    except Exception as e:
        print(f"Erreur zones: {e}")
        zones = []

    # Construction analysis
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
        'horizon_angle': round(float(horizon_angle), 2),
        'zones': zones
    }

    # ✅ DOUBLE CAPTION : Hugging Face API + Fallback
    caption_hf = None
    caption_fallback = None
    
    # 1. Hugging Face API
    try:
        print("🤗 Tentative Hugging Face API...")
        # Essayer BLIP d'abord (plus rapide que BLIP-2)
        caption_hf, error = generate_caption_huggingface(img_rgb, model_name='blip')
        
        if error:
            print(f"  ⚠️ Hugging Face échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur Hugging Face: {e}")
    
    # 2. Fallback
    caption_fallback = generate_smart_caption(analysis)
    
    # Sélectionner la meilleure
    if caption_hf:
        caption_main = caption_hf
    else:
        caption_main = caption_fallback
    
    # Stocker les captions
    analysis['caption'] = caption_main
    analysis['caption_huggingface'] = caption_hf or "Non disponible"
    analysis['caption_fallback'] = caption_fallback

    # Qualité
    analysis['quality_score'] = compute_quality_score(analysis)

    # Styles
    analysis['style_affinities'] = compute_all_style_affinities(img_rgb, analysis)

    # Composition rules
    try:
        analyzer = CompositionAnalyzer()
        analysis['composition_rules'] = analyzer.analyze_all_rules(img_rgb, subjects, analysis)
    except Exception as e:
        print(f"Erreur composition rules: {e}")
        analysis['composition_rules'] = {}

    return analysis


# ==========================================
# ROUTES FLASK
# ==========================================

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Endpoint analyse"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Pas d\'image fournie'}), 400
        
        # Décoder image
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = image_data.replace('data:image/webp;base64,', '')
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Analyse d'une image {img_rgb.shape}...")
        analysis = analyze_image_from_array(img_rgb)
        
        response = {
            'success': True,
            'sharpness': analysis.get('sharpness', 0),
            'brightness': analysis.get('brightness', 0),
            'contrast': analysis.get('contrast', 0),
            'noise': analysis.get('noise', 0),
            'quality_score': analysis.get('quality_score', 0),
            'subjects': analysis.get('subjects', []),
            'style_affinities': analysis.get('style_affinities', {}),
            'composition_rules': analysis.get('composition_rules', {}),
            'composition_score': analysis.get('composition', {}).get('composition_score', 0),
            'best_style': extract_best_style(analysis),
            'ai_prompt': generate_ai_prompt(analysis),
            'advice': extract_advice(analysis),
            'full_analysis': make_json_safe(analysis)
        }
        
        response = json.loads(json.dumps(response, default=str))
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


def extract_best_style(analysis):
    """Extraire meilleur style"""
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


def generate_ai_prompt(analysis):
    """Générer prompt IA"""
    try:
        scene = analysis.get('scene', {})
        scene_type = scene.get('scene_type', 'unknown').lower()
        brightness = analysis.get('brightness', 128)
        subjects = analysis.get('subjects', [])
        faces = analysis.get('faces', [])
        best_style = extract_best_style(analysis)
        
        prompt_parts = []
        
        if len(faces) > 0:
            prompt_parts.append("portrait" if len(faces) == 1 else "group portrait")
        elif any(s.get('class') == 'person' for s in subjects):
            prompt_parts.append("photograph of person")
        else:
            prompt_parts.append("photograph")
        
        if best_style:
            prompt_parts.append(f"in the style of {best_style['label'].lower()}")
        
        if brightness < 80:
            prompt_parts.append("moody lighting")
        elif brightness > 180:
            prompt_parts.append("bright, golden hour lighting")
        else:
            prompt_parts.append("soft natural lighting")
        
        prompt = ", ".join(prompt_parts)
        prompt = prompt[0].upper() + prompt[1:] if prompt else "Professional photograph"
        prompt += ". 8k, professional photography"
        
        return prompt
    except:
        return "Professional photograph, natural lighting, 8k"


def extract_advice(analysis):
    """Extraire conseils"""
    advice = []
    
    brightness = analysis.get('brightness', 128)
    sharpness = analysis.get('sharpness', 100)
    noise = analysis.get('noise', 20)
    
    if sharpness < 100:
        advice.append("🔍 Augmente la netteté")
    
    if brightness < 80:
        advice.append("☀️ Photo sombre, augmente l'exposition")
    elif brightness > 180:
        advice.append("⚡ Photo surexposée")
    
    if noise > 40:
        advice.append("🔇 Réduis le bruit")
    
    if not advice:
        advice.append("✨ Excellente photo !")
    
    return advice[:5]


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'API running with Hugging Face', 'version': '2.1'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )