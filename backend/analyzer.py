#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer.py
Script principal d'analyse d'images photographiques
Version complète avec système de légende intelligent
"""

import os
import sys
import json
import uuid
import time
import argparse
import cv2
from pathlib import Path

from config import Config
from core import (
    load_image_rgb, extract_exif, make_json_safe,
    compute_brightness, compute_contrast, compute_sharpness,
    estimate_noise_luminance, normalize_score, compute_saturation
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
    predict_scene, blip_caption,
    compute_all_style_affinities, compute_quality_score
)
from reporting import (
    generate_html_report, annotate_image
)
from composition_rules import CompositionAnalyzer
from intelligent_advisor import IntelligentAdvisor, generate_ai_coach_report
from improved_captioning import caption_with_context


def analyze_image(path, auto_rotate=True, user_level="intermédiaire"):
    """
    Fonction principale d'analyse
    
    Args:
        path: Chemin vers l'image
        auto_rotate: Auto-rotation EXIF
        user_level: "débutant", "intermédiaire", "avancé"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image introuvable : {path}")
    
    print(f"\n📷 Analyse de : {Path(path).name}")
    print(f"👤 Niveau détecté : {user_level.title()}")
    t0 = time.time()
    
    # ========================= CHARGEMENT =========================
    img_rgb = load_image_rgb(path, auto_rotate=auto_rotate)
    exif = extract_exif(path)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # ========================= MÉTRIQUES DE BASE =========================
    print("  📊 Métriques techniques...")
    brightness = compute_brightness(img_rgb)
    contrast = compute_contrast(img_rgb)
    sharpness = compute_sharpness(img_rgb)
    noise = estimate_noise_luminance(img_rgb)
    saturation = compute_saturation(img_rgb)
    
    # ========================= DÉTECTION DES DÉFAUTS =========================
    print("  🔍 Détection des défauts...")
    is_blurry = sharpness < Config.BLUR_THRESHOLD
    motion_blur_detected, motion_blur_score = detect_motion_blur(img_rgb)
    vignette_flag, vign_score = detect_vignette_advanced(img_rgb)
    chrom_ab_flag, chrom_score = detect_chromatic_aberration(img_rgb)
    horizon_angle = detect_horizon_angle(gray)
    
    # ========================= COULEURS =========================
    print("  🎨 Couleurs dominantes...")
    colors = dominant_colors(img_rgb)
    
    # ========================= DÉTECTION D'OBJETS =========================
    print("  🎯 Détection d'objets...")
    subjects = detect_objects(img_rgb)
    faces = detect_faces_mediapipe(img_rgb)
    
    # ========================= COMPOSITION =========================
    print("  🖼️ Analyse de composition...")
    comp = analyze_composition(img_rgb, subjects)
    
    # ========================= SCÈNE =========================
    print("  🌍 Classification de scène...")
    scene = predict_scene(img_rgb)
    
    # ========================= LÉGENDE INTELLIGENTE (NOUVEAU) =========================
    print("  💬 Génération de légende intelligente...")
    
    # Obtenir la légende BLIP brute
    blip_result, _ = blip_caption(img_rgb)
    
    # Analyse complète avec contexte
    caption_complete = caption_with_context(
        img_rgb, 
        analysis={
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness,
            'scene': scene,
            'black_and_white': saturation < 10
        },
        subjects=subjects,
        faces=faces,
        blip_caption=blip_result
    )
    
    caption = caption_complete['caption']
    caption_src = caption_complete['source']
    
    # Afficher les avertissements s'il y en a
    if caption_complete['warnings']:
        print("  ⚠️  Corrections appliquées :")
        for warning in caption_complete['warnings'][:2]:
            print(f"      • {warning}")
    
    # ========================= ZONES =========================
    print("  🗺️ Analyse par zones...")
    try:
        zones = build_zone_report(
            img_rgb, 
            subjects=subjects,
            scene_type=scene.get('scene_type', 'unknown')
        )
    except Exception as e:
        print(f"    ⚠️ Erreur zones : {e}")
        zones = []
    
    # ========================= CONSTRUCTION DU DICTIONNAIRE PRINCIPAL =========================
    analysis = {
        "path": path,
        "width": int(img_rgb.shape[1]),
        "height": int(img_rgb.shape[0]),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "sharpness": round(float(sharpness), 2),
        "is_blurry": bool(is_blurry),
        "motion_blur_detected": bool(motion_blur_detected),
        "motion_blur_score": round(float(motion_blur_score), 3),
        "noise": round(float(noise), 2),
        "dominant_colors": colors,
        "vignette": bool(vignette_flag),
        "vignette_score": round(float(vign_score), 3),
        "chrom_ab": (bool(chrom_ab_flag), round(float(chrom_score), 3)),
        "subjects": subjects,
        "faces": faces,
        "composition": comp,
        "exif": exif,
        "scene": scene,
        "caption": caption,
        "caption_source": caption_src,
        "caption_complete": caption_complete,
        "horizon_angle": round(float(horizon_angle), 2),
        "zones": zones
    }
    
    # ========================= SCORE DE QUALITÉ =========================
    quality = compute_quality_score(analysis)
    analysis["quality_score"] = quality
    
    # ========================= ANALYSE STYLISTIQUE AVANCÉE =========================
    print("  🎯 Analyse stylistique avancée...")
    style_affinities = compute_all_style_affinities(img_rgb, analysis)
    
    # ========================= GÉNÉRATION DES PRESETS DE RETOUCHE =========================
    print("  🎨 Génération des presets de retouche...")
    try:
        from preset_generator import generate_all_presets
        
        best_style = style_affinities.get("best_match")
        if best_style and best_style.get('score', 0) > 0.3:
            preset_files = generate_all_presets(best_style, analysis, output_dir="presets")
            analysis["preset_files"] = preset_files
            print(f"     ✅ Presets générés : {len(preset_files)} fichiers")
        else:
            analysis["preset_files"] = {}
    except Exception as e:
        print(f"     ⚠️  Erreur génération presets : {e}")
        analysis["preset_files"] = {}
    
    # ========================= ANALYSE DES RÈGLES DE COMPOSITION =========================
    print("  🖼️ Analyse des règles de composition...")
    try:
        composition_analyzer = CompositionAnalyzer()
        composition_rules_analysis = composition_analyzer.analyze_all_rules(img_rgb, subjects, analysis)
        
        # Obtenir le verdict d'éligibilité
        eligibility = composition_analyzer.get_eligibility_verdict(composition_rules_analysis)
        
        analysis["composition_rules"] = composition_rules_analysis
        analysis["composition_eligibility"] = eligibility
        
        print(f"\n📊 COMPOSITION : {composition_rules_analysis['summary']['grade']}")
        print(f"   Score global : {composition_rules_analysis['summary']['global_percentage']}%")
        print(f"   Règles respectées : {composition_rules_analysis['summary']['respect_rate']}")
        print(f"\n{eligibility['icon']} ÉLIGIBILITÉ : {eligibility['verdict']}")
        print(f"   {eligibility['message']}")
        print(f"   Temps estimé de retouche : {eligibility['estimated_work_time']}")
        
        if eligibility['priority_fixes']:
            print(f"\n💡 Suggestions prioritaires :")
            for i, sugg in enumerate(eligibility['priority_fixes'][:3], 1):
                print(f"   {i}. {sugg}")
    
    except Exception as e:
        print(f"    ⚠️ Erreur analyse composition : {e}")
        import traceback
        traceback.print_exc()
        analysis["composition_rules"] = None
        analysis["composition_eligibility"] = None
    
    # ========================= CONSEILS EXPERTS IA =========================
    print("  🎓 Génération conseils expert...")
    try:
        advisor = IntelligentAdvisor()
        expert_advice = advisor.analyze_and_advise(analysis, user_level=user_level)
        analysis["expert_advice"] = expert_advice
        
        # Génération du rapport coach
        coach_report = generate_ai_coach_report(analysis, user_level)
        analysis["coach_report"] = coach_report
        
    except Exception as e:
        print(f"    ⚠️ Erreur conseils experts : {e}")
        analysis["expert_advice"] = None
        analysis["coach_report"] = None
    
    # ========================= COUCHE V2 =========================
    analysis["v2"] = {
        "technical_analysis": {
            "brightness": normalize_score(brightness, *Config.BRIGHTNESS_RANGE),
            "contrast": normalize_score(contrast, *Config.CONTRAST_RANGE),
            "sharpness": normalize_score(sharpness, *Config.SHARPNESS_RANGE),
            "noise": 1 - normalize_score(noise, *Config.NOISE_RANGE)
        },
        "light_analysis": {
            "dramatic_light_score": normalize_score(contrast, 40, 120),
            "light_type": "natural"
        },
        "aesthetic_analysis": {
            "fine_art_score": normalize_score(sharpness + contrast, 0, 400)
        },
        "style_affinities": style_affinities
    }
    
    analysis["analysis_time_s"] = round(time.time() - t0, 3)
    
    # ========================= AFFICHAGE RÉSUMÉ TERMINAL =========================
    print(f"\n{'='*60}")
    print(f"✅ Analyse terminée en {analysis['analysis_time_s']}s")
    print(f"{'='*60}")
    print(f"📊 Score qualité : {quality}/100")
    print(f"📝 Légende : \"{caption}\" ({caption_src})")
    
    if style_affinities.get("best_match"):
        best = style_affinities["best_match"]
        print(f"🎨 Meilleur style : {best['label']} ({int(best['score']*100)}%)")
    
    # Affichage rapport coach
    if analysis.get("coach_report"):
        print("\n" + analysis["coach_report"])
    
    return analysis


def main():
    """Point d'entrée CLI"""
    parser = argparse.ArgumentParser(
        description="Analyseur d'images photographiques avec IA experte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python analyzer.py photo.jpg
  python analyzer.py photo.jpg --level débutant
  python analyzer.py photo.jpg --annotate
  python analyzer.py photo.jpg --output analysis.json --no-html
  python analyzer.py photo.jpg --styles custom_styles.json
        """
    )
    
    parser.add_argument("image", help="Chemin vers l'image à analyser")
    parser.add_argument("--level", default="intermédiaire",
                       choices=["débutant", "intermédiaire", "avancé"],
                       help="Votre niveau en photographie")
    parser.add_argument("--no-html", action="store_true",
                       help="Ne pas générer le rapport HTML")
    parser.add_argument("--annotate", action="store_true",
                       help="Générer l'image annotée")
    parser.add_argument("--output", default=None,
                       help="Nom du fichier JSON de sortie")
    parser.add_argument("--html-output", default=None,
                       help="Nom du rapport HTML personnalisé")
    parser.add_argument("--styles", default=None,
                       help="Chemin vers le fichier JSON des profils stylistiques")
    
    args = parser.parse_args()
    
    # Configuration profils
    if args.styles:
        Config.STYLES_PROFILES_FILE = args.styles
    
    try:
        # ========================= ANALYSE =========================
        analysis = analyze_image(args.image, user_level=args.level)
        
        # ========================= SAUVEGARDE JSON =========================
        if args.output:
            output_name = args.output
        else:
            image_name = Path(args.image).stem
            output_name = f"analysis_{image_name}.json"
        
        with open(output_name, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(analysis), f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON sauvegardé : {output_name}")
        
        # ========================= RAPPORT HTML =========================
        if not args.no_html:
            generate_html_report(analysis, out=args.html_output)
        
        # ========================= IMAGE ANNOTÉE =========================
        if args.annotate:
            annotate_image(args.image, analysis)
        
        # ========================= AFFICHAGE RÉSUMÉ DANS LE TERMINAL =========================
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE L'ANALYSE")
        print("="*60)
        print(f"Score qualité global : {analysis['quality_score']}/100")
        
        # Meilleur style
        best_match = analysis.get("v2", {}).get("style_affinities", {}).get("best_match")
        if best_match:
            print(f"\n🎨 Meilleur style : {best_match['label']}")
            print(f"   Score : {int(best_match['score']*100)}%")
            print(f"   Catégorie : {best_match['category']}")
            print(f"   Difficulté : {best_match['difficulty']}")
        
        # Top 3
        top_5 = analysis.get("v2", {}).get("style_affinities", {}).get("top_5", [])
        if len(top_5) > 1:
            print(f"\n🏆 Top 3 des styles compatibles :")
            for i, style in enumerate(top_5[:3], 1):
                print(f"   {i}. {style['label']} - {int(style['score']*100)}%")
        
        # Éligibilité composition
        eligibility = analysis.get("composition_eligibility")
        if eligibility:
            print(f"\n🖼️ RÈGLES DE COMPOSITION")
            print(f"   {eligibility['icon']} Verdict : {eligibility['verdict']}")
            print(f"   Score : {int(eligibility['global_score']*100)}%")
            print(f"   Éligibilité : {eligibility['eligibility']}")
            print(f"   Temps de retouche estimé : {eligibility['estimated_work_time']}")
            
            if eligibility['can_be_fixed'] and eligibility['priority_fixes']:
                print(f"\n   💡 Actions prioritaires :")
                for fix in eligibility['priority_fixes'][:3]:
                    print(f"      • {fix}")
        
        # Conseils experts
        expert_advice = analysis.get("expert_advice")
        if expert_advice:
            summary = expert_advice.get("summary", {})
            print(f"\n🎓 ANALYSE EXPERTE")
            print(f"   {summary.get('overall_assessment', 'N/A')}")
            
            if summary.get('next_steps'):
                print(f"\n   ⚡ Vos 3 prochaines actions :")
                for step in summary['next_steps']:
                    print(f"      {step}")
        
        # Légende
        caption_complete = analysis.get("caption_complete", {})
        if caption_complete:
            print(f"\n💬 LÉGENDE INTELLIGENTE")
            print(f"   Source : {caption_complete.get('source', 'N/A')}")
            print(f"   Confiance : {int(caption_complete.get('confidence', 0)*100)}%")
            print(f"   Texte : \"{analysis.get('caption', 'N/A')}\"")
        
        print("\n✅ Traitement terminé avec succès !")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()