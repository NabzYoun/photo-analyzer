# -*- coding: utf-8 -*-
"""
preset_generator.py
Génération de presets de retouche personnalisés
"""

import json
from pathlib import Path


def calculate_adjustments(current_values, target_values):
    """
    Calcule les ajustements nécessaires pour atteindre les valeurs cibles
    """
    adjustments = {}
    
    for metric, target in target_values.items():
        if isinstance(target, dict) and 'ideal' in target:
            current = current_values.get(metric, 0)
            ideal = target['ideal']
            
            # Calculer l'écart
            if metric == 'brightness':
                # Conversion en stops EV (approximation)
                if current < ideal:
                    ev_adjustment = (ideal - current) / 50  # ~50 points = 1 EV
                    adjustments['exposure'] = f"+{ev_adjustment:.1f} EV"
                else:
                    ev_adjustment = (current - ideal) / 50
                    adjustments['exposure'] = f"-{ev_adjustment:.1f} EV"
            
            elif metric == 'contrast':
                if current < ideal:
                    adjustments['contrast'] = f"+{int((ideal - current) * 2)}"
                else:
                    adjustments['contrast'] = f"-{int((current - ideal) * 2)}"
            
            elif metric == 'saturation':
                if current < ideal:
                    adjustments['vibrance'] = f"+{int((ideal - current) / 2)}"
                else:
                    adjustments['vibrance'] = f"-{int((current - ideal) / 2)}"
    
    return adjustments


def generate_lightroom_preset(style_profile, current_analysis):
    """
    Génère un preset Lightroom adapté
    """
    base_preset = style_profile.get('lightroom_preset', {})
    target_values = style_profile.get('target_values', {})
    current_values = {
        'brightness': current_analysis.get('brightness', 0),
        'contrast': current_analysis.get('contrast', 0),
        'saturation': current_analysis.get('saturation', 0)
    }
    
    # Calculer les ajustements personnalisés
    adjustments = calculate_adjustments(current_values, target_values)
    
    # Fusionner avec le preset de base
    custom_preset = base_preset.copy()
    
    # Ajuster exposure si nécessaire
    if 'exposure' in adjustments:
        ev_str = adjustments['exposure'].replace(' EV', '').replace('+', '')
        try:
            ev_delta = float(ev_str)
            custom_preset['exposure'] = base_preset.get('exposure', 0) + ev_delta
        except:
            pass
    
    return custom_preset


def generate_step_by_step_guide(style_profile, current_analysis):
    """
    Génère un guide pas-à-pas personnalisé
    """
    steps = []
    current_brightness = current_analysis.get('brightness', 0)
    current_contrast = current_analysis.get('contrast', 0)
    current_saturation = current_analysis.get('saturation', 0)
    
    target_values = style_profile.get('target_values', {})
    
    # Étape 1 : Exposition
    if 'brightness' in target_values:
        target_brightness = target_values['brightness'].get('ideal', 0)
        if abs(current_brightness - target_brightness) > 20:
            ev_adjustment = (target_brightness - current_brightness) / 50
            steps.append({
                "step": len(steps) + 1,
                "category": "Exposition",
                "action": f"Ajuster l'exposition de {ev_adjustment:+.1f} EV",
                "tool": "Curseur Exposure",
                "priority": "high",
                "reason": f"Votre photo est à {current_brightness:.0f}, cible: {target_brightness:.0f}"
            })
    
    # Étape 2 : Contraste
    if 'contrast' in target_values:
        target_contrast = target_values['contrast'].get('ideal', 0)
        if abs(current_contrast - target_contrast) > 10:
            contrast_adjustment = int((target_contrast - current_contrast) * 2)
            steps.append({
                "step": len(steps) + 1,
                "category": "Contraste",
                "action": f"Ajuster le contraste de {contrast_adjustment:+d}",
                "tool": "Curseur Contrast",
                "priority": "high",
                "reason": f"Contraste actuel: {current_contrast:.0f}, cible: {target_contrast:.0f}"
            })
    
    # Étape 3 : Ombres et hautes lumières
    if 'shadows' in target_values:
        shadows_adj = target_values['shadows'].get('adjustment', '')
        priority = target_values['shadows'].get('priority', 'medium')
        steps.append({
            "step": len(steps) + 1,
            "category": "Ombres",
            "action": f"Régler Shadows à {shadows_adj}",
            "tool": "Curseur Shadows",
            "priority": priority,
            "reason": "Essentiel pour le style " + style_profile.get('label', '')
        })
    
    if 'highlights' in target_values:
        highlights_adj = target_values['highlights'].get('adjustment', '')
        priority = target_values['highlights'].get('priority', 'medium')
        steps.append({
            "step": len(steps) + 1,
            "category": "Hautes lumières",
            "action": f"Régler Highlights à {highlights_adj}",
            "tool": "Curseur Highlights",
            "priority": priority,
            "reason": "Pour conserver les détails"
        })
    
    # Étape 4 : Blancs et noirs
    if 'whites' in target_values:
        steps.append({
            "step": len(steps) + 1,
            "category": "Blancs",
            "action": f"Régler Whites à {target_values['whites'].get('adjustment', '')}",
            "tool": "Curseur Whites",
            "priority": target_values['whites'].get('priority', 'medium'),
            "reason": "Ajuster la luminosité globale"
        })
    
    if 'blacks' in target_values:
        steps.append({
            "step": len(steps) + 1,
            "category": "Noirs",
            "action": f"Régler Blacks à {target_values['blacks'].get('adjustment', '')}",
            "tool": "Curseur Blacks",
            "priority": target_values['blacks'].get('priority', 'medium'),
            "reason": "Définir le point noir"
        })
    
    # Étape 5 : Clarté et texture
    if 'clarity' in target_values:
        steps.append({
            "step": len(steps) + 1,
            "category": "Clarté",
            "action": f"Régler Clarity à {target_values['clarity'].get('adjustment', '')}",
            "tool": "Curseur Clarity",
            "priority": target_values['clarity'].get('priority', 'low'),
            "reason": "Ajuster le micro-contraste"
        })
    
    # Étape 6 : Couleurs
    if 'vibrance' in target_values:
        steps.append({
            "step": len(steps) + 1,
            "category": "Couleurs",
            "action": f"Régler Vibrance à {target_values['vibrance'].get('adjustment', '')}",
            "tool": "Curseur Vibrance",
            "priority": target_values['vibrance'].get('priority', 'medium'),
            "reason": "Ajuster l'intensité des couleurs"
        })
    
    return steps


def generate_gap_analysis(style_profile, current_analysis):
    """
    Analyse l'écart entre l'état actuel et le style cible
    """
    gaps = []
    target_values = style_profile.get('target_values', {})
    
    for metric, target in target_values.items():
        if isinstance(target, dict) and 'ideal' in target:
            current = current_analysis.get(metric, 0)
            ideal = target['ideal']
            min_val = target.get('min', ideal - 20)
            max_val = target.get('max', ideal + 20)
            
            status = "ok"
            if current < min_val:
                status = "too_low"
                gap_amount = min_val - current
            elif current > max_val:
                status = "too_high"
                gap_amount = current - max_val
            else:
                gap_amount = 0
            
            gaps.append({
                "metric": metric,
                "current": round(current, 1),
                "target": ideal,
                "range": f"{min_val}-{max_val}",
                "status": status,
                "gap": round(gap_amount, 1),
                "unit": target.get('unit', '')
            })
    
    return gaps


def generate_preset_files(style_profile, current_analysis, output_dir="presets"):
    """
    Génère les fichiers de preset dans différents formats
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    style_id = style_profile.get('id', 'unknown')
    
    # 1. Lightroom XMP
    lr_preset = generate_lightroom_preset(style_profile, current_analysis)
    lr_file = Path(output_dir) / f"{style_id}_lightroom.json"
    with open(lr_file, 'w', encoding='utf-8') as f:
        json.dump(lr_preset, f, indent=2)
    
    # 2. Guide étape par étape
    steps = generate_step_by_step_guide(style_profile, current_analysis)
    guide_file = Path(output_dir) / f"{style_id}_guide.json"
    with open(guide_file, 'w', encoding='utf-8') as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)
    
    # 3. Analyse des écarts
    gaps = generate_gap_analysis(style_profile, current_analysis)
    gap_file = Path(output_dir) / f"{style_id}_gaps.json"
    with open(gap_file, 'w', encoding='utf-8') as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)
    
    return {
        "lightroom_preset": str(lr_file),
        "step_by_step_guide": str(guide_file),
        "gap_analysis": str(gap_file)
    }


def generate_markdown_guide(style_profile, current_analysis, steps):
    """
    Génère un guide Markdown lisible
    """
    md = f"""# Guide de retouche : {style_profile.get('label', 'Style')}

## 📊 État actuel de votre photo

- **Luminosité** : {current_analysis.get('brightness', 0):.0f} / 255
- **Contraste** : {current_analysis.get('contrast', 0):.0f}
- **Saturation** : {current_analysis.get('saturation', 0):.0f}
- **Netteté** : {current_analysis.get('sharpness', 0):.0f}

## 🎯 Objectif du style "{style_profile.get('label', '')}"

{style_profile.get('description', '')}

**Caractéristiques clés :**
"""
    
    for char in style_profile.get('key_characteristics', []):
        md += f"- {char}\n"
    
    md += "\n## 📝 Étapes de retouche\n\n"
    
    for step in steps:
        priority_icon = "🔴" if step['priority'] == 'critical' else "🟠" if step['priority'] == 'high' else "🟡" if step['priority'] == 'medium' else "🟢"
        md += f"""### {priority_icon} Étape {step['step']} : {step['category']}

**Action** : {step['action']}  
**Outil** : {step['tool']}  
**Pourquoi** : {step['reason']}

"""
    
    md += "\n## 💡 Conseils supplémentaires\n\n"
    
    for tip in style_profile.get('pro_tips', []):
        md += f"- {tip}\n"
    
    md += "\n## ⚠️ Erreurs à éviter\n\n"
    
    for mistake in style_profile.get('common_mistakes', []):
        md += f"- ❌ {mistake}\n"
    
    return md


# Fonction principale d'intégration
def generate_all_presets(style_profile, current_analysis, output_dir="presets"):
    """
    Génère tous les formats de presets et guides
    """
    # Générer les fichiers JSON
    files = generate_preset_files(style_profile, current_analysis, output_dir)
    
    # Générer le guide Markdown
    steps = generate_step_by_step_guide(style_profile, current_analysis)
    md_guide = generate_markdown_guide(style_profile, current_analysis, steps)
    
    md_file = Path(output_dir) / f"{style_profile.get('id', 'style')}_guide.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_guide)
    
    files['markdown_guide'] = str(md_file)
    
    return files
