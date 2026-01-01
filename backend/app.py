"""
app.py - Backend Flask avec Google Vision API + BLIP-2
Installation : 
pip install flask flask-cors python-dotenv
pip install google-cloud-vision
pip install transformers pillow torch
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

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import modules analyzer
from analyzer import analyze_image
from core import make_json_safe

# ==========================================
# GOOGLE VISION API
# ==========================================

from google.cloud import vision
from google.oauth2 import service_account

_vision_client = None

def init_vision_client():
    """Initialise Google Vision API"""
    global _vision_client
    
    if _vision_client is not None:
        return True
    
    try:
        # Chemin vers credentials JSON
        credentials_path = os.path.join(BASE_DIR, 'google-vision-credentials.json')
        
        if not os.path.exists(credentials_path):
            print(f"⚠️ Credentials non trouvés: {credentials_path}")
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
    """Génère une description avec Google Vision API"""
    try:
        if not init_vision_client():
            return None, "Vision API non initialisée"
        
        print("  🔍 Génération Google Vision...")
        
        # Convertir numpy array en bytes
        img_pil = Image.fromarray(img_rgb)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG', quality=85)
        content = img_byte_arr.getvalue()
        
        image = vision.Image(content=content)
        
        # Détections multiples
        label_response = _vision_client.label_detection(image=image)
        labels = [label.description for label in label_response.label_annotations[:8]]
        
        face_response = _vision_client.face_detection(image=image)
        faces = face_response.face_annotations
        
        text_response = _vision_client.text_detection(image=image)
        has_text = len(text_response.text_annotations) > 0
        
        web_response = _vision_client.web_detection(image=image)
        web_entities = [e.description for e in web_response.web_entities[:3] if e.description]
        
        # Construire description française
        caption = build_french_caption_from_vision(labels, faces, has_text, web_entities)
        
        print(f"  ✅ Google Vision: {caption}")
        return caption, None
        
    except Exception as e:
        print(f"  ❌ Erreur Google Vision: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def build_french_caption_from_vision(labels, faces, has_text, web_entities):
    """Construit une description française depuis Google Vision"""
    translations = {
        'person': 'personne', 'man': 'homme', 'woman': 'femme', 'child': 'enfant',
        'dog': 'chien', 'cat': 'chat', 'hat': 'chapeau', 'cap': 'casquette',
        'glasses': 'lunettes', 'sunglasses': 'lunettes de soleil', 'smile': 'souriant',
        'building': 'bâtiment', 'street': 'rue', 'city': 'ville', 'urban': 'urbain',
        'nature': 'nature', 'sky': 'ciel', 'tree': 'arbre', 'car': 'voiture',
        'outdoor': 'en extérieur', 'indoor': 'en intérieur', 'landscape': 'paysage',
        'mountain': 'montagne', 'beach': 'plage', 'forest': 'forêt', 'water': 'eau',
        'wall': 'mur', 'standing': 'debout', 'sitting': 'assis', 'clothing': 'vêtements',
        'shirt': 'chemise', 'jacket': 'veste', 'phone': 'téléphone', 'table': 'table'
    }
    
    labels_fr = [translations.get(l.lower(), l.lower()) for l in labels[:5]]
    
    parts = []
    
    # Analyse des visages
    if faces:
        nb = len(faces)
        face = faces[0]
        
        likelihood_scores = {'VERY_UNLIKELY': 0, 'UNLIKELY': 1, 'POSSIBLE': 2, 'LIKELY': 3, 'VERY_LIKELY': 4}
        
        emotions = {
            'joy': likelihood_scores.get(str(face.joy_likelihood).split('.')[-1], 0),
            'sorrow': likelihood_scores.get(str(face.sorrow_likelihood).split('.')[-1], 0),
            'anger': likelihood_scores.get(str(face.anger_likelihood).split('.')[-1], 0),
            'surprise': likelihood_scores.get(str(face.surprise_likelihood).split('.')[-1], 0)
        }
        
        dominant = max(emotions, key=emotions.get)
        score = emotions[dominant]
        
        if nb == 1:
            if 'homme' in labels_fr:
                parts.append("Un homme")
            elif 'femme' in labels_fr:
                parts.append("Une femme")
            else:
                parts.append("Une personne")
            
            if score >= 3:
                emotion_fr = {'joy': 'souriant', 'sorrow': 'triste', 'anger': 'sérieux', 'surprise': 'surpris'}
                parts.append(emotion_fr.get(dominant, ''))
        else:
            parts.append(f"{nb} personnes")
    
    # Accessoires
    accessories = [l for l in labels_fr if l in ['casquette', 'chapeau', 'lunettes', 'lunettes de soleil']]
    if accessories:
        if len(accessories) == 1:
            parts.append(f"portant une {accessories[0]}")
        else:
            parts.append(f"portant {' et '.join(accessories)}")
    
    # Contexte
    contexts = {'rue': 'dans une rue', 'ville': 'en ville', 'urbain': 'en milieu urbain',
                'nature': 'dans la nature', 'paysage': 'dans un paysage', 
                'plage': 'à la plage', 'montagne': 'en montagne', 'forêt': 'en forêt'}
    
    for key, val in contexts.items():
        if key in labels_fr:
            parts.append(val)
            break
    
    # Texte visible
    if has_text:
        parts.append("avec du texte visible")
    
    if not parts or (len(parts) == 1 and len(parts[0]) < 10):
        caption = f"Photo montrant {', '.join(labels_fr[:3])}" if labels_fr else "Photographie"
    else:
        caption = ' '.join(parts)
    
    caption = caption[0].upper() + caption[1:] if caption else "Photographie"
    if not caption.endswith('.'):
        caption += '.'
    
    return caption


# ==========================================
# BLIP-2
# ==========================================

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    print("⚠️ transformers ou torch non installé, BLIP-2 désactivé")

_blip2_model = None
_blip2_processor = None

def load_blip2_model():
    """Charge BLIP-2 une seule fois"""
    global _blip2_model, _blip2_processor
    
    if not BLIP2_AVAILABLE:
        return False
    
    if _blip2_model is None:
        try:
            print("🔄 Chargement BLIP-2 (peut prendre 1-2 minutes)...")
            
            model_name = "Salesforce/blip2-opt-2.7b"
            
            _blip2_processor = Blip2Processor.from_pretrained(model_name)
            _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                _blip2_model = _blip2_model.cuda()
                print("✅ BLIP-2 chargé sur GPU")
            else:
                print("✅ BLIP-2 chargé sur CPU (peut être lent)")
                
            return True
        except Exception as e:
            print(f"❌ Erreur chargement BLIP-2: {e}")
            return False
    
    return True


def generate_caption_blip2(img_rgb):
    """Génère une description avec BLIP-2"""
    try:
        if not load_blip2_model():
            return None, "BLIP-2 non disponible"
        
        print("  🎨 Génération BLIP-2...")
        
        img_pil = Image.fromarray(img_rgb)
        inputs = _blip2_processor(img_pil, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = _blip2_model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7
            )
        
        caption_en = _blip2_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        caption_fr = translate_simple(caption_en)
        
        print(f"  ✅ BLIP-2: {caption_fr}")
        return caption_fr, None
        
    except Exception as e:
        print(f"  ❌ Erreur BLIP-2: {e}")
        return None, str(e)


def translate_simple(english_text):
    """Traduction simple anglais -> français"""
    trans = {
        'a man': 'un homme', 'a woman': 'une femme', 'a person': 'une personne',
        'wearing': 'portant', 'a hat': 'un chapeau', 'a cap': 'une casquette',
        'glasses': 'des lunettes', 'smiling': 'souriant', 'standing': 'debout',
        'sitting': 'assis', 'in front of': 'devant', 'behind': 'derrière',
        'with': 'avec', 'and': 'et', 'red': 'rouge', 'blue': 'bleu',
        'black': 'noir', 'white': 'blanc', 'a wall': 'un mur',
        'a building': 'un bâtiment', 'a car': 'une voiture', 'outside': 'à l\'extérieur',
        'looking at': 'regardant', 'holding': 'tenant'
    }
    
    text_fr = english_text.lower()
    for en, fr in trans.items():
        text_fr = text_fr.replace(en, fr)
    
    return text_fr[0].upper() + text_fr[1:] if text_fr else english_text


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Endpoint analyse avec double caption"""
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


def generate_smart_caption(analysis):
    """Fallback intelligent basé sur l'analyse"""
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
        
        if len(faces) > 0:
            parts.append('avec un portrait' if len(faces) == 1 else f'avec {len(faces)} personnes')
        elif subjects:
            subject_classes = [s['class'] for s in subjects[:3]]
            if 'person' in subject_classes:
                parts.append('montrant une ou plusieurs personnes')
            else:
                translations = {
                    'car': 'une voiture', 'dog': 'un chien', 'cat': 'un chat',
                    'tree': 'des arbres', 'building': 'des bâtiments'
                }
                translated = [translations.get(s, s) for s in subject_classes]
                if len(translated) == 1:
                    parts.append(f'avec {translated[0]}')
                elif len(translated) == 2:
                    parts.append(f'avec {translated[0]} et {translated[1]}')
                else:
                    parts.append(f'avec {", ".join(translated[:2])} et plus')
        
        if brightness < 80:
            parts.append('dans une ambiance sombre')
        elif brightness > 180:
            parts.append('très lumineuse')
        else:
            parts.append('bien éclairée')
        
        caption = ' '.join(parts) + '.'
        caption = caption[0].upper() + caption[1:]
        
        return caption
        
    except Exception as e:
        print(f"Erreur generate_smart_caption: {e}")
        return "Photographie professionnelle analysée."


def analyze_image_from_array(img_rgb):
    """Analyse image avec DOUBLE CAPTION (Google Vision + BLIP-2)"""
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

    # ✅ DOUBLE CAPTION : Google Vision + BLIP-2
    caption_google = None
    caption_blip2 = None
    caption_fallback = None
    
    # 1. Google Vision
    try:
        print("📊 Tentative Google Vision...")
        caption_google, error = generate_caption_google_vision(img_rgb)
        if error:
            print(f"  ⚠️ Google Vision échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur Google Vision: {e}")
    
    # 2. BLIP-2
    try:
        print("🎨 Tentative BLIP-2...")
        caption_blip2, error = generate_caption_blip2(img_rgb)
        if error:
            print(f"  ⚠️ BLIP-2 échoué: {error}")
    except Exception as e:
        print(f"  ❌ Erreur BLIP-2: {e}")
    
    # 3. Fallback intelligent
    caption_fallback = generate_smart_caption(analysis)
    
    # Sélectionner la meilleure caption pour 'caption' principal
    if caption_google:
        caption_main = caption_google
    elif caption_blip2:
        caption_main = caption_blip2
    else:
        caption_main = caption_fallback
    
    # Stocker toutes les captions
    analysis['caption'] = caption_main
    analysis['caption_google'] = caption_google or "Non disponible"
    analysis['caption_blip2'] = caption_blip2 or "Non disponible"
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


def extract_best_style(analysis):
    """Extraire le meilleur style"""
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
            prompt_parts.append("moody lighting, dark atmosphere")
        elif brightness > 180:
            prompt_parts.append("bright, well-lit, golden hour lighting")
        else:
            prompt_parts.append("soft natural lighting")
        
        prompt = ", ".join(prompt_parts)
        prompt = prompt[0].upper() + prompt[1:] if prompt else "Professional photograph"
        prompt += ". 8k, professional photography"
        
        return prompt
    
    except Exception as e:
        print(f"Erreur prompt: {e}")
        return "Professional photograph, natural lighting, sharp details, 8k"


def extract_advice(analysis):
    """Extraire conseils"""
    advice = []
    
    brightness = analysis.get('brightness', 128)
    sharpness = analysis.get('sharpness', 100)
    noise = analysis.get('noise', 20)
    
    if sharpness < 100:
        advice.append("🔍 Augmente la netteté avec Clarity")
    
    if brightness < 80:
        advice.append("☀️ Photo sombre, augmente l'exposition")
    elif brightness > 180:
        advice.append("⚡ Photo surexposée, réduis les highlights")
    
    if noise > 40:
        advice.append("🔇 Bruit élevé, utilise la réduction de bruit")
    
    if not advice:
        advice.append("✨ Excellente photo technique !")
    
    return advice[:5]


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'API running', 'version': '2.0 - Dual Caption'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )