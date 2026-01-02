"""
app.py - Version finale avec Hugging Face API fonctionnelle
Installation : pip install requests pillow
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

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analyzer import analyze_image
from core import make_json_safe

app = Flask(__name__)
CORS(app)


# ==========================================
# HUGGING FACE API - MODÈLE QUI FONCTIONNE
# ==========================================

HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")

# Modèles testés et fonctionnels
WORKING_MODELS = {
    'nlpconnect': 'nlpconnect/vit-gpt2-image-captioning',  # Rapide et fiable
    'microsoft': 'microsoft/git-base',  # Alternative
}


def generate_caption_huggingface(img_rgb, model_key='nlpconnect'):
    """
    Génère description avec Hugging Face API
    Utilise un modèle qui fonctionne réellement
    """
    try:
        token = HUGGINGFACE_API_TOKEN.strip() if HUGGINGFACE_API_TOKEN else None
        
        if not token or token == "hf_VotreTokenIci":
            print("  ⚠️ Token Hugging Face non configuré")
            return None, "Token non configuré"
        
        print(f"  🤗 Génération Hugging Face ({model_key})...")
        
        # Réduire l'image pour éviter timeout
        img_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_pil.thumbnail((512, 512))  # Plus petit = plus rapide
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=80)
        img_bytes = buffered.getvalue()
        
        # Modèle qui fonctionne
        model_id = WORKING_MODELS[model_key]
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {
            "Authorization": f"Bearer {token}",
        }
        
        print(f"  🚀 Requête vers : {api_url}")
        
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        
        print(f"  📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Parser selon le format
            caption_en = None
            if isinstance(result, list) and len(result) > 0:
                caption_en = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                caption_en = result.get('generated_text', '')
            
            if caption_en:
                caption_en = caption_en.strip()
                caption_fr = translate_advanced(caption_en)
                
                print(f"  ✅ HF: {caption_fr}")
                return caption_fr, None
            else:
                print(f"  ⚠️ Format réponse inattendu: {result}")
                return None, "Format inattendu"
            
        elif response.status_code == 503:
            print("  ⏳ Modèle en chargement (attendez 20s et réessayez)")
            return None, "Modèle en chargement"
            
        elif response.status_code == 401:
            print("  ❌ Token invalide ou expiré")
            return None, "Token invalide"
            
        else:
            error_text = response.text[:300]
            print(f"  ❌ Erreur {response.status_code}: {error_text}")
            return None, f"Erreur {response.status_code}"
        
    except requests.exceptions.Timeout:
        print("  ⏰ Timeout (réessayez)")
        return None, "Timeout"
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")
        return None, str(e)


def translate_advanced(english_text):
    """
    Traduction améliorée anglais -> français
    """
    # Nettoyage
    text = english_text.lower().strip()
    
    # Dictionnaire enrichi
    translations = {
        # Personnes
        'a man': 'un homme',
        'a woman': 'une femme',
        'a person': 'une personne',
        'a boy': 'un garçon',
        'a girl': 'une fille',
        'people': 'des personnes',
        'two people': 'deux personnes',
        'group of people': 'un groupe de personnes',
        
        # Vêtements
        'wearing': 'portant',
        'a hat': 'un chapeau',
        'a cap': 'une casquette',
        'glasses': 'des lunettes',
        'sunglasses': 'des lunettes de soleil',
        'a shirt': 'une chemise',
        'a jacket': 'une veste',
        'a suit': 'un costume',
        'a dress': 'une robe',
        'a tie': 'une cravate',
        
        # Actions
        'smiling': 'souriant',
        'standing': 'debout',
        'sitting': 'assis',
        'walking': 'marchant',
        'looking at': 'regardant',
        'holding': 'tenant',
        'posing': 'posant',
        'playing': 'jouant',
        
        # Lieux
        'in front of': 'devant',
        'behind': 'derrière',
        'next to': 'à côté de',
        'a wall': 'un mur',
        'a building': 'un bâtiment',
        'a house': 'une maison',
        'a street': 'une rue',
        'the street': 'la rue',
        'outside': 'à l\'extérieur',
        'inside': 'à l\'intérieur',
        'on the beach': 'sur la plage',
        'in the park': 'dans le parc',
        
        # Objets
        'a car': 'une voiture',
        'a bicycle': 'un vélo',
        'a tree': 'un arbre',
        'trees': 'des arbres',
        'the sky': 'le ciel',
        'a phone': 'un téléphone',
        'a camera': 'un appareil photo',
        'a table': 'une table',
        'a chair': 'une chaise',
        'a dog': 'un chien',
        'a cat': 'un chat',
        
        # Couleurs
        'red': 'rouge',
        'blue': 'bleu',
        'green': 'vert',
        'yellow': 'jaune',
        'black': 'noir',
        'white': 'blanc',
        'gray': 'gris',
        
        # Connecteurs
        'with': 'avec',
        'and': 'et',
        'on': 'sur',
        'of': 'de',
        'in': 'dans',
        'at': 'à',
    }
    
    # Appliquer les traductions
    text_fr = text
    for en, fr in translations.items():
        text_fr = text_fr.replace(en, fr)
    
    # Capitaliser
    if text_fr:
        text_fr = text_fr[0].upper() + text_fr[1:]
    
    return text_fr if text_fr else english_text


# ==========================================
# FALLBACK INTELLIGENT
# ==========================================

def generate_smart_caption(analysis):
    """Génère description basée sur détections"""
    try:
        subjects = analysis.get('subjects', [])
        scene = analysis.get('scene', {})
        scene_type = scene.get('scene_type', 'unknown')
        brightness = analysis.get('brightness', 128)
        faces = analysis.get('faces', [])
        
        parts = []
        
        # Type de scène
        scene_map = {
            'mountain': 'Une photo de montagne',
            'forest': 'Une photo en forêt',
            'beach': 'Une photo de plage',
            'urban': 'Une photo urbaine',
            'street': 'Une photo de rue',
            'indoor': 'Une photo en intérieur',
            'landscape': 'Un paysage',
            'sky': 'Une photo avec ciel'
        }
        
        for key, desc in scene_map.items():
            if key in scene_type.lower():
                parts.append(desc)
                break
        
        if not parts:
            parts.append('Une photographie')
        
        # Visages/personnes
        if len(faces) > 0:
            if len(faces) == 1:
                parts.append('montrant un portrait')
            else:
                parts.append(f'avec {len(faces)} personnes')
        elif subjects:
            subject_classes = [s['class'] for s in subjects[:3]]
            if 'person' in subject_classes:
                parts.append('avec une ou plusieurs personnes')
            else:
                obj_trans = {
                    'car': 'une voiture', 'dog': 'un chien', 'cat': 'un chat',
                    'tree': 'des arbres', 'building': 'un bâtiment'
                }
                translated = [obj_trans.get(s, s) for s in subject_classes[:2]]
                if translated:
                    parts.append(f"montrant {' et '.join(translated)}")
        
        # Ambiance
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
        return "Photographie analysée."


# ==========================================
# ANALYSE IMAGE
# ==========================================

def analyze_image_from_array(img_rgb):
    """Analyse avec double caption : HuggingFace + Fallback"""
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
        zones = []

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

    # ✅ DOUBLE CAPTION
    caption_hf = None
    caption_fallback = None
    
    # 1. Hugging Face
    try:
        print("🤗 Tentative Hugging Face...")
        caption_hf, error = generate_caption_huggingface(img_rgb, model_key='nlpconnect')
        if error:
            print(f"  ⚠️ HF échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur HF: {e}")
    
    # 2. Fallback
    caption_fallback = generate_smart_caption(analysis)
    
    # Sélectionner
    caption_main = caption_hf if caption_hf else caption_fallback
    
    analysis['caption'] = caption_main
    analysis['caption_huggingface'] = caption_hf or "Non disponible"
    analysis['caption_fallback'] = caption_fallback

    # Reste
    analysis['quality_score'] = compute_quality_score(analysis)
    analysis['style_affinities'] = compute_all_style_affinities(img_rgb, analysis)

    try:
        analyzer = CompositionAnalyzer()
        analysis['composition_rules'] = analyzer.analyze_all_rules(img_rgb, subjects, analysis)
    except:
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
            return jsonify({'error': 'Pas d\'image'}), 400
        
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = image_data.replace('data:image/webp;base64,', '')
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Analyse {img_rgb.shape}...")
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
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


def extract_best_style(analysis):
    try:
        best = analysis.get('style_affinities', {}).get('best_match')
        if best:
            return {
                'label': best.get('label', 'Unknown'),
                'description': best.get('description', ''),
                'score': round(best.get('score', 0), 3),
                'category': best.get('category', ''),
                'difficulty': best.get('difficulty', '')
            }
    except:
        pass
    return None


def generate_ai_prompt(analysis):
    try:
        brightness = analysis.get('brightness', 128)
        subjects = analysis.get('subjects', [])
        faces = analysis.get('faces', [])
        style = extract_best_style(analysis)
        
        parts = []
        
        if faces:
            parts.append("portrait" if len(faces) == 1 else "group portrait")
        elif any(s.get('class') == 'person' for s in subjects):
            parts.append("photograph of person")
        else:
            parts.append("photograph")
        
        if style:
            parts.append(f"in style of {style['label'].lower()}")
        
        if brightness < 80:
            parts.append("moody lighting")
        elif brightness > 180:
            parts.append("golden hour")
        
        return ", ".join(parts) + ". 8k, professional"
    except:
        return "Professional photograph, 8k"


def extract_advice(analysis):
    advice = []
    b = analysis.get('brightness', 128)
    s = analysis.get('sharpness', 100)
    n = analysis.get('noise', 20)
    
    if s < 100: advice.append("🔍 Augmente netteté")
    if b < 80: advice.append("☀️ Augmente exposition")
    elif b > 180: advice.append("⚡ Réduis exposition")
    if n > 40: advice.append("🔇 Réduis bruit")
    if not advice: advice.append("✨ Excellente photo!")
    
    return advice[:5]


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'Photo Analyzer API', 'version': '4.0'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )