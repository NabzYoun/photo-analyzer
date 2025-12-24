# -*- coding: utf-8 -*-
"""
config.py
Configuration centralisée pour l'analyseur d'images
"""

class Config:
    """Configuration globale de l'application"""
    
    # Traitement d'images
    MAX_SIZE = 1400
    YOLO_IMG_SIZE = 320
    
    # Seuils de détection
    BLUR_THRESHOLD = 80.0
    MOTION_BLUR_THRESHOLD = 15.0
    VIGNETTE_THRESHOLD = 0.06
    CHROMATIC_AB_THRESHOLD = 0.08
    
    # Fichiers de données
    PLACES_CLASSES_FILE = "data/categories_places365.txt"
    PLACES_MODEL_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    STYLES_PROFILES_FILE = "data/styles_profiles.json"
    
    # Plages de normalisation
    BRIGHTNESS_RANGE = (60, 200)
    CONTRAST_RANGE = (20, 100)
    SHARPNESS_RANGE = (80, 300)
    NOISE_RANGE = (10, 50)
    SATURATION_RANGE = (20, 150)
    DYNAMIC_RANGE_RANGE = (50, 200)
    
    # Segmentation
    SEGMENT_SIZE = 200
    N_CLUSTERS = 3
    
    # Couleurs
    N_DOMINANT_COLORS = 4
    
    # Rapports
    DEFAULT_HTML_OUTPUT = "rapport_analyse_{}.html"
    DEFAULT_JSON_OUTPUT = "analysis_{}.json"
    DEFAULT_ANNOTATED_OUTPUT = "annotated_{}.jpg"


class ModelCache:
    """Cache pour les modèles chargés"""
    
    def __init__(self):
        self._cache = {
            'yolo': None,
            'blip_processor': None,
            'blip_model': None,
            'places_model': None,
            'places_classes': None,
            'style_profiles': None
        }
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
    
    def clear(self, key=None):
        if key:
            self._cache[key] = None
        else:
            self._cache = {k: None for k in self._cache}


# Instance globale du cache
model_cache = ModelCache()
