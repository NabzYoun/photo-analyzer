"""
app.py - Avec Google Vision API + Hugging Face API
Installation : pip install requests google-cloud-vision
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
from dotenv import load_dotenv
import io
import requests

# Charger .env
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analyzer import analyze_image
from core import make_json_safe

# Importer Google Vision (optionnel)
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("⚠️ Google Cloud Vision non installé")

app = Flask(__name__)
CORS(app)

# Variables globales
_vision_client = None


# ==========================================
# GOOGLE VISION API
# ==========================================

def init_google_vision():
    """Initialise Google Vision API"""
    global _vision_client
    
    if not GOOGLE_VISION_AVAILABLE:
        return False
    
    if _vision_client is not None:
        return True
    
    try:
        credentials_path = os.path.join(BASE_DIR, 'google-vision-credentials.json')
        
        if not os.path.exists(credentials_path):
            print(f"⚠️ Credentials Google non trouvés: {credentials_path}")
            return False
        
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        
        _vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        print("✅ Google Vision API initialisée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur init Google Vision: {e}")
        return False


def generate_caption_google_vision(img_rgb):
    """Génère description avec Google Vision API"""
    try:
        if not init_google_vision():
            return None, "Google Vision non initialisée"
        
        print("  🔍 Génération Google Vision...")
        
        img_pil = Image.fromarray(img_rgb)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG', quality=85)
        content = img_byte_arr.getvalue()
        
        image = vision.Image(content=content)
        
        # Détections
        label_response = _vision_client.label_detection(image=image)
        labels = [label.description for label in label_response.label_annotations[:8]]
        
        face_response = _vision_client.face_detection(image=image)
        faces = face_response.face_annotations
        
        # Construction description
        caption = build_caption_from_google(labels, faces)
        
        print(f"  ✅ Google Vision: {caption}")
        return caption, None
        
    except Exception as e:
        print(f"  ❌ Erreur Google Vision: {e}")
        return None, str(e)


def build_caption_from_google(labels, faces):
    """Construit description depuis Google Vision"""
    translations = {
        'person': 'personne', 'man': 'homme', 'woman': 'femme',
        'dog': 'chien', 'cat': 'chat', 'cap': 'casquette',
        'glasses': 'lunettes', 'smile': 'souriant', 'building': 'bâtiment',
        'street': 'rue', 'city': 'ville', 'car': 'voiture',
        'tree': 'arbre', 'sky': 'ciel', 'nature': 'nature'
    }
    
    labels_fr = [translations.get(l.lower(), l.lower()) for l in labels[:5]]
    parts = []
    
    # Visages
    if faces:
        nb = len(faces)
        if nb == 1:
            if 'homme' in labels_fr:
                parts.append("Un homme")
            elif 'femme' in labels_fr:
                parts.append("Une femme")
            else:
                parts.append("Une personne")
            
            # Vérifier émotion
            face = faces[0]
            likelihood_scores = {
                'VERY_UNLIKELY': 0, 'UNLIKELY': 1, 'POSSIBLE': 2,
                'LIKELY': 3, 'VERY_LIKELY': 4
            }
            joy_score = likelihood_scores.get(str(face.joy_likelihood).split('.')[-1], 0)
            if joy_score >= 3:
                parts.append("souriant")
        else:
            parts.append(f"{nb} personnes")
    
    # Accessoires
    accessories = [l for l in labels_fr if l in ['casquette', 'chapeau', 'lunettes']]
    if accessories:
        parts.append(f"portant une {accessories[0]}")
    
    # Contexte
    if 'rue' in labels_fr or 'ville' in labels_fr:
        parts.append("dans un environnement urbain")
    elif 'nature' in labels_fr:
        parts.append("dans la nature")
    
    if not parts:
        return f"Photo montrant {', '.join(labels_fr[:3])}." if labels_fr else "Photographie."
    
    caption = ' '.join(parts)
    caption = caption[0].upper() + caption[1:] if caption else "Photographie"
    
    if not caption.endswith('.'):
        caption += '.'
    
    return caption


# ==========================================
# HUGGING FACE API (URL CORRIGÉE)
# ==========================================

HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")


def generate_caption_huggingface(img_rgb, model_name='blip'):
    """
    Génère description avec Hugging Face API
    URL corrigée pour éviter 404
    """
    try:
        token = HUGGINGFACE_API_TOKEN.strip() if HUGGINGFACE_API_TOKEN else None
        
        if not token or token == "hf_VotreTokenIci":
            print("  ⚠️ Token Hugging Face non configuré")
            return None, "Token non configuré"
        
        print(f"  🤗 Génération Hugging Face ({model_name})...")
        
        # Réduire l'image
        img_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_pil.thumbnail((800, 800))
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        
        # ✅ URL CORRIGÉE - API Inference officielle
        model_id = "Salesforce/blip-image-captioning-large"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {
            "Authorization": f"Bearer {token}",
        }
        
        print(f"  🚀 Requête vers : {api_url}")
        
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        
        print(f"  DEBUG STATUS: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Parser la réponse
            if isinstance(result, list) and len(result) > 0:
                caption_en = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                caption_en = result.get('generated_text', '')
            else:
                caption_en = str(result)
            
            caption_en = caption_en.strip()
            caption_fr = translate_simple(caption_en)
            
            print(f"  ✅ Hugging Face: {caption_fr}")
            return caption_fr, None
            
        elif response.status_code == 503:
            print("  ⏳ Modèle en chargement, réessayez dans 20s")
            return None, "Modèle en chargement"
            
        else:
            print(f"  DEBUG MSG: {response.text[:200]}")
            return None, f"Erreur {response.status_code}"
        
    except requests.exceptions.Timeout:
        print("  ❌ Timeout")
        return None, "Timeout"
    except Exception as e:
        print(f"  ❌ Erreur: {str(e)}")
        return None, str(e)


def translate_simple(english_text):
    """Traduction simple anglais -> français"""
    trans = {
        'a man': 'un homme', 'a woman': 'une femme', 'a person': 'une personne',
        'wearing': 'portant', 'a cap': 'une casquette', 'a hat': 'un chapeau',
        'glasses': 'des lunettes', 'sunglasses': 'lunettes de soleil',
        'smiling': 'souriant', 'standing': 'debout', 'sitting': 'assis',
        'in front of': 'devant', 'behind': 'derrière', 'next to': 'à côté de',
        'with': 'avec', 'and': 'et', 'red': 'rouge', 'blue': 'bleu',
        'black': 'noir', 'white': 'blanc', 'a wall': 'un mur',
        'a building': 'un bâtiment', 'a car': 'une voiture',
        'a tree': 'un arbre', 'the sky': 'le ciel',
        'looking at': 'regardant', 'holding': 'tenant'
    }
    
    text_fr = english_text.lower()
    for en, fr in trans.items():
        text_fr = text_fr.replace(en, fr)
    
    return text_fr[0].upper() + text_fr[1:] if text_fr else english_text


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
        
        scene_descriptions = {
            'mountain': 'Une photo de montagne',
            'forest': 'Une photo en forêt',
            'beach': 'Une photo de plage',
            'urban': 'Une photo urbaine',
            'street': 'Une photo de rue',
            'indoor': 'Une photo en intérieur',
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
        
        if len(faces) > 0:
            parts.append('avec un portrait' if len(faces) == 1 else f'avec {len(faces)} personnes')
        elif subjects:
            subject_classes = [s['class'] for s in subjects[:3]]
            if 'person' in subject_classes:
                parts.append('montrant une ou plusieurs personnes')
        
        if brightness < 80:
            parts.append('dans une ambiance sombre')
        elif brightness > 180:
            parts.append('très lumineuse')
        else:
            parts.append('bien éclairée')
        
        caption = ' '.join(parts) + '.'
        return caption[0].upper() + caption[1:]
        
    except Exception as e:
        print(f"Erreur fallback: {e}")
        return "Photographie professionnelle."


# ==========================================
# ANALYSE IMAGE
# ==========================================

def analyze_image_from_array(img_rgb):
    """Analyse avec TRIPLE caption : Google + HuggingFace + Fallback"""
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

    # ✅ TRIPLE CAPTION
    caption_google = None
    caption_hf = None
    caption_fallback = None
    
    # 1. Google Vision
    try:
        print("🔍 Tentative Google Vision...")
        caption_google, error = generate_caption_google_vision(img_rgb)
        if error:
            print(f"  ⚠️ Google Vision échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur Google Vision: {e}")
    
    # 2. Hugging Face
    try:
        print("🤗 Tentative Hugging Face...")
        caption_hf, error = generate_caption_huggingface(img_rgb)
        if error:
            print(f"  ⚠️ Hugging Face échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur Hugging Face: {e}")
    
    # 3. Fallback
    caption_fallback = generate_smart_caption(analysis)
    
    # Sélectionner la meilleure
    if caption_google:
        caption_main = caption_google
    elif caption_hf:
        caption_main = caption_hf
    else:
        caption_main = caption_fallback
    
    # Stocker toutes les captions
    analysis['caption'] = caption_main
    analysis['caption_google'] = caption_google or "Non disponible"
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
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Pas d\'image fournie'}), 400
        
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
    try:
        scene = analysis.get('scene', {})
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
            prompt_parts.append("golden hour lighting")
        
        prompt = ", ".join(prompt_parts) + ". 8k, professional photography"
        return prompt[0].upper() + prompt[1:]
    except:
        return "Professional photograph, 8k"


def extract_advice(analysis):
    advice = []
    
    brightness = analysis.get('brightness', 128)
    sharpness = analysis.get('sharpness', 100)
    noise = analysis.get('noise', 20)
    
    if sharpness < 100:
        advice.append("🔍 Augmente la netteté")
    
    if brightness < 80:
        advice.append("☀️ Augmente l'exposition")
    elif brightness > 180:
        advice.append("⚡ Réduis l'exposition")
    
    if noise > 40:
        advice.append("🔇 Réduis le bruit")
    
    if not advice:
        advice.append("✨ Excellente photo !")
    
    return advice[:5]


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'API Google Vision + Hugging Face', 'version': '3.0'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )