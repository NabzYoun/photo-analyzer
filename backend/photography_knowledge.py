# -*- coding: utf-8 -*-
"""
photography_knowledge.py
Base de connaissances photographiques contextuelles
"""

from typing import Dict, List, Optional


class PhotographyKnowledge:
    """
    Base de connaissances photographiques avec contexte
    """
    
    # ========================= LUMIÈRE =========================
    LIGHT_SCENARIOS = {
        "golden_hour": {
            "time": ["sunrise +/- 1h", "sunset +/- 1h"],
            "characteristics": ["Lumière chaude", "Ombres douces", "Contraste doux"],
            "best_for": ["Portraits", "Paysages", "Architecture"],
            "avoid": ["Contraste élevé", "Flash"],
            "iso_range": (100, 400),
            "tips": [
                "Photographiez avec le soleil derrière vous pour lumière dorée",
                "Ou à contre-jour pour silhouettes/halos",
                "C'est la lumière magique, utilisez-la !",
                "Planifiez vos sorties 1h avant coucher du soleil"
            ]
        },
        "blue_hour": {
            "time": ["30min après coucher du soleil", "30min avant lever"],
            "characteristics": ["Lumière bleue froide", "Faible intensité", "Ambiance mystérieuse"],
            "best_for": ["Cityscapes", "Architecture éclairée", "Long exposure"],
            "avoid": ["Portraits sans lumière d'appoint"],
            "iso_range": (800, 3200),
            "tips": [
                "Trépied obligatoire",
                "Bracketing pour HDR recommandé",
                "Équilibre parfait entre ciel et lumières artificielles",
                "Durée très courte : soyez prêt à l'avance"
            ]
        },
        "harsh_midday": {
            "time": ["11h-15h"],
            "characteristics": ["Ombres dures", "Contraste élevé", "Lumière plate"],
            "best_for": ["Architecture graphique", "N&B contrasté"],
            "avoid": ["Portraits", "Paysages naturels"],
            "iso_range": (100, 200),
            "tips": [
                "Cherchez l'ombre pour les portraits",
                "Utilisez réflecteur ou fill-flash",
                "Ou exploitez le contraste en N&B",
                "Idéal pour graphisme architectural"
            ]
        },
        "overcast": {
            "time": ["Jour nuageux"],
            "characteristics": ["Lumière diffuse", "Pas d'ombres", "Couleurs saturées"],
            "best_for": ["Portraits", "Macro", "Nature", "Couleurs vives"],
            "avoid": ["Contrastes forts", "Silhouettes"],
            "iso_range": (200, 800),
            "tips": [
                "Lumière parfaite pour portraits naturels",
                "Aucune ombre disgracieuse",
                "Les couleurs ressortent mieux",
                "Pas de problème de hautes lumières cramées"
            ]
        },
        "indoor_natural": {
            "characteristics": ["Lumière de fenêtre", "Directionnel", "Contraste modéré"],
            "best_for": ["Portraits intimistes", "Still life", "Produits"],
            "iso_range": (400, 1600),
            "tips": [
                "Fenêtre nord = lumière constante",
                "Utilisez voilages pour diffuser",
                "Réflecteur blanc pour déboucher ombres",
                "Évitez lumière directe du soleil"
            ]
        },
        "indoor_artificial": {
            "characteristics": ["Lumière mixte", "Balance des blancs complexe", "ISO élevé"],
            "best_for": ["Événements", "Concerts", "Restaurants"],
            "iso_range": (1600, 6400),
            "tips": [
                "Réglez balance des blancs sur auto ou custom",
                "ISO élevé + ouverture max nécessaires",
                "Post-traitement crucial pour température couleur",
                "Flash externe avec diffuseur si autorisé"
            ]
        }
    }
    
    # ========================= GENRES PHOTOGRAPHIQUES =========================
    GENRE_REQUIREMENTS = {
        "portrait": {
            "focal_length_ideal": (50, 135),
            "aperture_ideal": (1.4, 2.8),
            "shutter_speed_min": "1/125s",
            "iso_max": 1600,
            "composition_rules": ["Règle des tiers", "Espace négatif", "Profondeur de champ"],
            "light_preference": ["Lumière douce", "Golden hour", "Fenêtre nord"],
            "critical_factors": [
                "Les YEUX doivent être nets (priorité absolue)",
                "Catchlight dans les yeux pour vie",
                "Éviter distorsion (pas de grand angle)",
                "Fond flou pour isoler le sujet",
                "Expression naturelle > perfection technique"
            ],
            "common_mistakes": [
                "Flash direct en pleine face (ombres dures)",
                "Trop de profondeur de champ (f/11+)",
                "Sujet au centre (statique)",
                "Couper aux articulations (genoux, coudes)",
                "Négliger l'arrière-plan"
            ]
        },
        "landscape": {
            "focal_length_ideal": (16, 35),
            "aperture_ideal": (8, 16),
            "shutter_speed_range": "Variable",
            "iso_max": 400,
            "composition_rules": ["Règle des tiers", "Lignes directrices", "Premier plan"],
            "light_preference": ["Golden hour", "Blue hour", "Overcast"],
            "critical_factors": [
                "Netteté sur toute l'image (hyperfocale)",
                "Premier plan pour profondeur",
                "Ciel intéressant (nuages, couleurs)",
                "Patience pour la bonne lumière",
                "Trépied quasi-obligatoire"
            ],
            "common_mistakes": [
                "Horizon au centre (divise l'image en 2)",
                "Pas de premier plan (manque de profondeur)",
                "Ciel cramé (sous-exposer ou HDR)",
                "Grande ouverture (f/2.8 = pas assez net)",
                "Photographier à midi (lumière plate)"
            ]
        },
        "street": {
            "focal_length_ideal": (28, 50),
            "aperture_ideal": (5.6, 8),
            "shutter_speed_min": "1/250s",
            "iso_auto": True,
            "composition_rules": ["Instant décisif", "Juxtaposition", "Géométrie"],
            "light_preference": ["Toute lumière (adaptabilité)"],
            "critical_factors": [
                "Anticiper l'action",
                "Discrétion (petit appareil)",
                "Zone de netteté pré-réglée (zone focus)",
                "Composition rapide",
                "L'émotion > la technique"
            ],
            "common_mistakes": [
                "Hésiter et rater le moment",
                "Trop près sans consentement",
                "Photos floues par manque de vitesse",
                "Composition négligée",
                "Peur d'augmenter l'ISO"
            ]
        },
        "wildlife": {
            "focal_length_ideal": (300, 600),
            "aperture_ideal": (4, 5.6),
            "shutter_speed_min": "1/1000s",
            "iso_max": 3200,
            "composition_rules": ["Espace devant regard", "Niveau des yeux", "Bokeh"],
            "light_preference": ["Golden hour", "Overcast"],
            "critical_factors": [
                "Vitesse élevée pour figer mouvement",
                "Rafale + AF continu obligatoire",
                "Distance de sécurité respectée",
                "Patience extrême",
                "Yeux de l'animal nets (priorité)"
            ],
            "common_mistakes": [
                "Vitesse trop lente (flou de bougé)",
                "Photographier de trop loin (sujet perdu)",
                "Couper les oreilles/queue",
                "Ignorer le comportement animal",
                "Fond distrayant"
            ]
        },
        "macro": {
            "focal_length_ideal": (60, 105),
            "aperture_ideal": (11, 16),
            "shutter_speed_range": "Variable (trépied)",
            "iso_max": 800,
            "composition_rules": ["Simplicité", "Abstraction", "Symétrie"],
            "light_preference": ["Lumière diffuse", "Overcast"],
            "critical_factors": [
                "Trépied obligatoire",
                "Focus stacking pour profondeur",
                "Stabilité absolue nécessaire",
                "Lumière douce (flash macro/annulaire)",
                "Patience pour insectes"
            ],
            "common_mistakes": [
                "Profondeur de champ trop faible",
                "Flou de bougé invisible à l'œil",
                "Lumière dure (ombres disgracieuses)",
                "Arrière-plan chargé",
                "Respiration qui bouge l'appareil"
            ]
        }
    }
    
    # ========================= INTENTIONS CRÉATIVES =========================
    CREATIVE_INTENTIONS = {
        "bokeh_prononcé": {
            "technique": "Grande ouverture + longue focale",
            "settings": {"aperture": "f/1.4-f/2.8", "focal": "85mm+", "distance": "Proche sujet"},
            "purpose": "Isoler sujet, créer rêve, flou artistique",
            "tips": [
                "Distance sujet-fond importante",
                "Plus la focale est longue, plus le bokeh est prononcé",
                "Qualité du bokeh = qualité de l'objectif",
                "Attention : zone de netteté très faible !"
            ]
        },
        "effet_filé": {
            "technique": "Panning + vitesse lente",
            "settings": {"shutter": "1/30s - 1/125s", "mode": "Priorité vitesse"},
            "purpose": "Dynamisme, vitesse, énergie",
            "tips": [
                "Suivre le sujet en mouvement fluide",
                "Plusieurs essais nécessaires",
                "Fond flou = vitesse, sujet net = maîtrise",
                "Idéal pour sports, voitures, vélos"
            ]
        },
        "light_painting": {
            "technique": "Pose longue + source lumineuse mobile",
            "settings": {"shutter": "10s+", "aperture": "f/8-f/11", "iso": "100-400"},
            "purpose": "Créativité, art, effet spectaculaire",
            "tips": [
                "Nuit noire obligatoire",
                "Trépied + déclencheur à distance",
                "Portez des vêtements sombres (invisibilité)",
                "Lampe LED/torche pour dessiner"
            ]
        },
        "silhouette": {
            "technique": "Contre-jour + sous-exposition",
            "settings": {"exposure": "-1 à -2 EV", "metering": "Spot sur ciel"},
            "purpose": "Mystère, graphisme, minimalisme",
            "tips": [
                "Sujet reconnaissable par sa forme",
                "Lever/coucher de soleil idéal",
                "Exposition sur fond lumineux",
                "Formes simples et épurées"
            ]
        },
        "hdr_naturel": {
            "technique": "Bracketing + fusion",
            "settings": {"bracketing": "3-5 photos", "ev_step": "+/- 2 EV"},
            "purpose": "Conserver détails ombres ET lumières",
            "tips": [
                "Trépied obligatoire",
                "Scènes à fort contraste (intérieur/fenêtre)",
                "Éviter l'effet HDR exagéré (fake)",
                "Fusion subtile en post-production"
            ]
        }
    }
    
    # ========================= ERREURS COURANTES PAR NIVEAU =========================
    COMMON_MISTAKES_BY_LEVEL = {
        "débutant": [
            {
                "mistake": "Photographier en JPEG au lieu de RAW",
                "why_bad": "Perte d'informations irréversible, marge de retouche limitée",
                "fix": "Toujours shooter en RAW pour flexibilité maximale",
                "impact": "critical"
            },
            {
                "mistake": "Utiliser le flash intégré en direct",
                "why_bad": "Lumière dure, yeux rouges, ombres disgracieuses",
                "fix": "Augmenter ISO ou utiliser flash externe avec diffuseur",
                "impact": "high"
            },
            {
                "mistake": "Mettre le sujet au centre systématiquement",
                "why_bad": "Composition statique et ennuyeuse",
                "fix": "Règle des tiers, nombre d'or, asymétrie intentionnelle",
                "impact": "medium"
            },
            {
                "mistake": "Photographier en Auto total",
                "why_bad": "Aucun contrôle créatif, l'appareil décide pour vous",
                "fix": "Passer en mode A/Av (priorité ouverture) minimum",
                "impact": "high"
            },
            {
                "mistake": "Négliger l'arrière-plan",
                "why_bad": "Éléments distrayants qui ruinent la photo",
                "fix": "Toujours vérifier le fond avant de déclencher",
                "impact": "high"
            }
        ],
        "intermédiaire": [
            {
                "mistake": "Sur-traiter les photos (HDR, clarté, saturation)",
                "why_bad": "Aspect artificiel, halos, perte de naturel",
                "fix": "Retouche subtile, moins c'est plus",
                "impact": "high"
            },
            {
                "mistake": "Négliger la lumière",
                "why_bad": "Lumière = 80% de la photo, technique = 20%",
                "fix": "Attendre/chercher la bonne lumière plutôt que forcer",
                "impact": "critical"
            },
            {
                "mistake": "Copier les styles à la mode sans réfléchir",
                "why_bad": "Photos sans personnalité, manque d'authenticité",
                "fix": "Développer son propre style, expérimenter",
                "impact": "medium"
            },
            {
                "mistake": "Shooter trop en rafale sans réfléchir",
                "why_bad": "Perte du sens de la composition, triage fastidieux",
                "fix": "Ralentir, composer, 1 bonne photo > 100 moyennes",
                "impact": "medium"
            }
        ],
        "avancé": [
            {
                "mistake": "Négliger la pré-visualisation",
                "why_bad": "Manque d'intention claire, photos sans âme",
                "fix": "Visualiser le résultat avant de shooter (Ansel Adams)",
                "impact": "high"
            },
            {
                "mistake": "Technique > Émotion",
                "why_bad": "Photos techniquement parfaites mais froides",
                "fix": "L'émotion et le message > la perfection technique",
                "impact": "critical"
            },
            {
                "mistake": "Ne pas assumer ses choix artistiques",
                "why_bad": "Photos tièdes qui plaisent à personne",
                "fix": "Affirmez votre vision, même si polarisante",
                "impact": "medium"
            }
        ]
    }
    
    def get_contextual_advice(self, 
                             genre: str,
                             light_condition: str,
                             subject_type: str,
                             user_level: str = "intermédiaire") -> Dict:
        """
        Génère des conseils contextuels basés sur la situation
        """
        
        advice = {
            "genre_tips": [],
            "light_tips": [],
            "common_mistakes": [],
            "settings_recommendation": {},
            "creative_opportunities": []
        }
        
        # Conseils genre
        if genre in self.GENRE_REQUIREMENTS:
            genre_data = self.GENRE_REQUIREMENTS[genre]
            advice["genre_tips"] = genre_data.get("critical_factors", [])
            advice["common_mistakes"].extend(genre_data.get("common_mistakes", []))
            advice["settings_recommendation"] = {
                "focal_length": genre_data.get("focal_length_ideal"),
                "aperture": genre_data.get("aperture_ideal"),
                "iso_max": genre_data.get("iso_max")
            }
        
        # Conseils lumière
        if light_condition in self.LIGHT_SCENARIOS:
            light_data = self.LIGHT_SCENARIOS[light_condition]
            advice["light_tips"] = light_data.get("tips", [])
            advice["settings_recommendation"]["iso_range"] = light_data.get("iso_range")
        
        # Erreurs courantes du niveau
        if user_level in self.COMMON_MISTAKES_BY_LEVEL:
            advice["common_mistakes"].extend([
                m["mistake"] for m in self.COMMON_MISTAKES_BY_LEVEL[user_level][:3]
            ])
        
        return advice


# Fonction utilitaire
def detect_light_condition(analysis: Dict) -> str:
    """Détecte la condition lumineuse depuis l'analyse"""
    brightness = analysis.get('brightness', 128)
    contrast = analysis.get('contrast', 50)
    scene = analysis.get('scene', {}).get('scene_type', '')
    
    # Indoor
    if 'room' in scene.lower() or 'indoor' in scene.lower():
        return 'indoor_natural' if brightness > 100 else 'indoor_artificial'
    
    # Outdoor
    if brightness > 180:
        return 'harsh_midday'
    elif brightness < 80:
        return 'blue_hour'
    elif 120 < brightness < 160 and contrast < 40:
        return 'overcast'
    elif contrast > 30:
        return 'golden_hour'
    
    return 'overcast'


def detect_genre(analysis: Dict) -> str:
    """Détecte le genre photographique"""
    subjects = analysis.get('subjects', [])
    scene = analysis.get('scene', {}).get('scene_type', '')
    has_face = len(analysis.get('faces', [])) > 0
    
    # Wildlife
    if any(s.get('class') in ['cat', 'dog', 'bird', 'horse'] for s in subjects):
        return 'wildlife'
    
    # Portrait
    if has_face:
        return 'portrait'
    
    # Landscape
    if any(kw in scene.lower() for kw in ['mountain', 'field', 'forest', 'ocean', 'sky']):
        return 'landscape'
    
    # Street
    if any(kw in scene.lower() for kw in ['street', 'urban', 'city']):
        return 'street'
    
    # Macro
    if analysis.get('sharpness', 0) > 200 and len(subjects) <= 2:
        return 'macro'
    
    return 'general'