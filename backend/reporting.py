# -*- coding: utf-8 -*-
"""
reporting.py
Génération de rapports HTML complets avec toutes les analyses
"""

import json
import cv2
from pathlib import Path
from core import make_json_safe


def get_color_for_score(score):
    """Retourne une couleur selon le score (0-1)"""
    if score >= 0.8:
        return "#4CAF50"
    elif score >= 0.6:
        return "#8BC34A"
    elif score >= 0.4:
        return "#FF9800"
    else:
        return "#F44336"


def build_expert_advice_html(analysis):
    """Construit la section HTML des conseils experts"""
    
    expert_advice = analysis.get("expert_advice")
    
    if not expert_advice:
        return ""
    
    context = expert_advice.get("context", {})
    summary = expert_advice.get("summary", {})
    immediate_fixes = expert_advice.get("immediate_fixes", [])
    shooting_improvements = expert_advice.get("shooting_improvements", [])
    creative_suggestions = expert_advice.get("creative_suggestions", [])
    learning_resources = expert_advice.get("learning_resources", [])
    
    html = f"""
    <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:25px; border-radius:10px; margin:30px 0; color:white;">
        <h2 style="color:white; margin-top:0;">🎓 Analyse par un Photographe Expert</h2>
        
        <div style="background:rgba(255,255,255,0.1); padding:15px; border-radius:8px; margin:15px 0;">
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:15px; font-size:0.95em;">
                <div><strong>📸 Genre :</strong><br>{context.get('genre', 'N/A').title()}</div>
                <div><strong>💡 Lumière :</strong><br>{context.get('light_condition', 'N/A').replace('_', ' ').title()}</div>
                <div><strong>👤 Niveau :</strong><br>{context.get('user_level', 'N/A').title()}</div>
            </div>
        </div>
        
        <div style="background:rgba(255,255,255,0.15); padding:20px; border-radius:8px; margin:15px 0; text-align:center; font-size:1.1em;">
            {summary.get('overall_assessment', 'Analyse en cours...')}
        </div>
    </div>
    """
    
    if immediate_fixes:
        html += '<div style="background:white; padding:20px; border-radius:10px; margin:20px 0; border-left:5px solid #f44336;">'
        html += '<h3 style="color:#f44336; margin-top:0;">🔧 Corrections Immédiates</h3>'
        
        for i, fix in enumerate(immediate_fixes[:5], 1):
            priority_color = "#f44336" if fix.get('priority') == 'high' else "#ff9800"
            html += f"""
            <div style="background:#f9f9f9; padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid {priority_color};">
                <h4 style="margin:0 0 10px 0; color:#333;">{i}. {fix.get('issue', 'N/A')}</h4>
                <div style="background:white; padding:12px; border-radius:5px;">
                    <strong style="color:#4CAF50;">✓ Solution :</strong> {fix.get('fix', 'N/A')}
                </div>
                <div style="color:#666; margin-top:8px; font-size:0.9em;">
                    <strong>🛠️ Outil :</strong> <code>{fix.get('tool', 'N/A')}</code>
                </div>
            </div>
            """
        html += '</div>'
    
    if shooting_improvements:
        html += '<div style="background:white; padding:20px; border-radius:10px; margin:20px 0; border-left:5px solid #2196F3;">'
        html += '<h3 style="color:#2196F3; margin-top:0;">📸 Pour Vos Prochaines Photos</h3>'
        
        for i, imp in enumerate(shooting_improvements[:5], 1):
            html += f"""
            <div style="background:#f0f8ff; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h4 style="margin:0 0 10px 0; color:#333;">{imp.get('issue', imp.get('solution', 'N/A'))}</h4>
            """
            if 'solution' in imp:
                html += f'<div style="color:#2196F3;">→ {imp["solution"]}</div>'
            html += '</div>'
        
        html += '</div>'
    
    return html


def build_composition_rules_html(analysis):
    """Construit la section HTML des règles de composition"""
    
    comp_rules = analysis.get("composition_rules")
    eligibility = analysis.get("composition_eligibility")
    
    if not comp_rules or not eligibility:
        return ""
    
    rules = comp_rules.get("rules", {})
    summary = comp_rules.get("summary", {})
    
    html = f"""
    <div style="background:#f0f8ff; padding:20px; border-radius:10px; margin:30px 0; border-left:5px solid #2196F3;">
        <h2 style="color:#2196F3; margin-top:0;">📐 Règles de Composition</h2>
        
        <div style="background:white; padding:20px; border-radius:8px; margin:20px 0; text-align:center;">
            <h3 style="margin:0; color:#333;">Score Global</h3>
            <div style="font-size:3em; font-weight:bold; color:{get_color_for_score(summary.get('global_score', 0))}; margin:10px 0;">
                {summary.get('global_percentage', 0)}%
            </div>
            <div style="font-size:1.5em; color:#666;">{summary.get('grade', 'N/A')}</div>
            <div style="color:#888; font-size:0.9em;">{summary.get('respected_count', 0)}/{summary.get('total_count', 0)} règles respectées</div>
        </div>
        
        <div style="background:{eligibility.get('bg_color', '#fff')}; border-left:4px solid {eligibility.get('color', '#ccc')}; padding:20px; border-radius:8px;">
            <h3 style="color:{eligibility.get('color', '#333')};">{eligibility.get('icon', '❓')} {eligibility.get('verdict', 'N/A')}</h3>
            <p style="color:#333;">{eligibility.get('message', '')}</p>
        </div>
        
        <h3 style="color:#333; margin-top:30px;">📋 Détail par règle</h3>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:15px;">
    """
    
    for rule_name, rule_data in rules.items():
        score_percent = int(rule_data.get('score', 0) * 100)
        color = get_color_for_score(rule_data.get('score', 0))
        suggestions = '<br>'.join(rule_data.get('suggestions', [])[:2])
        
        html += f"""
        <div style="background:white; padding:15px; border-radius:8px; border-left:4px solid {color};">
            <h4 style="margin:0 0 10px 0; color:#333;">{rule_name}</h4>
            <div style="font-size:1.5em; font-weight:bold; color:{color}; margin-bottom:10px;">{score_percent}%</div>
            <div style="background:#f9f9f9; padding:10px; border-radius:5px; font-size:0.85em;">
                <strong>💡 Suggestion :</strong><br>{suggestions if suggestions else 'Aucune suggestion'}
            </div>
        </div>
        """
    
    html += '</div></div>'
    return html


def build_pro_summary(analysis):
    """Construction tableau de synthèse"""
    summary = []
    
    sharp = analysis.get("sharpness", 0)
    if sharp < 50:
        summary.append(("Netteté", "Floue", "Augmentez vitesse d'obturation"))
    elif sharp < 150:
        summary.append(("Netteté", "Moyenne", "Utilisez un trépied"))
    else:
        summary.append(("Netteté", "Très bonne", "RAS"))
    
    if analysis.get("motion_blur_detected"):
        summary.append(("Flou de mouvement", "Présent", "Vitesse plus rapide"))
    
    noise = analysis.get("noise", 0)
    if noise < 20:
        summary.append(("Bruit", "Faible", "RAS"))
    elif noise < 40:
        summary.append(("Bruit", "Modéré", "Réduisez ISO"))
    else:
        summary.append(("Bruit", "Élevé", "Réduction de bruit"))
    
    bright = analysis.get("brightness", 0)
    if bright < 80:
        summary.append(("Exposition", "Sous-exposée", "Augmenter exposition"))
    elif bright > 180:
        summary.append(("Exposition", "Surexposée", "Réduire exposition"))
    else:
        summary.append(("Exposition", "Correcte", "RAS"))
    
    cont = analysis.get("contrast", 0)
    if cont < 30:
        summary.append(("Contraste", "Faible", "Augmenter"))
    elif cont > 80:
        summary.append(("Contraste", "Élevé", "Réduire"))
    else:
        summary.append(("Contraste", "Correct", "RAS"))
    
    if analysis.get("vignette"):
        summary.append(("Vignetage", "Présent", "Correction profil"))
    
    angle = abs(analysis.get("horizon_angle", 0))
    if angle > 3:
        summary.append(("Horizon", f"Penché ({angle:.1f}°)", "Redresser"))
    else:
        summary.append(("Horizon", "Droit", "RAS"))
    
    faces = analysis.get("faces", [])
    if faces:
        summary.append(("Visages", f"{len(faces)} détecté(s)", "Vérifier netteté"))
    
    subjects = analysis.get("subjects", [])
    summary.append(("Objets", str(len(subjects)), "—"))
    
    scene = analysis.get("scene", {})
    summary.append(("Scène", scene.get("scene_type", "N/A"), "—"))
    
    quality = analysis.get("quality_score")
    if quality:
        summary.append(("Score qualité", f"{quality}/100", "—"))
    
    return summary


def generate_html_report(analysis, out=None):
    """Génération rapport HTML COMPLET"""
    if out is None:
        image_path = analysis.get("path", "image")
        image_name = Path(image_path).stem
        out = f"rapport_analyse_{image_name}.html"
    
    safe = make_json_safe(analysis)
    rows = build_pro_summary(analysis)
    
    # EXIF
    exif_html = ""
    for k, v in (safe.get("exif") or {}).items():
        if v:
            exif_html += f"<li><b>{k}:</b> {v}</li>"
    
    # Zones
    zones_html = ""
    for z in safe.get("zones", []):
        zones_html += f"""
        <li>
            <b>{z.get('label','zone')}</b> – {z.get('percent_area',0)}% – 
            lum={z.get('brightness',0)}, sat={z.get('saturation',0)}, sharp={z.get('sharpness',0)}
            <br><span style="color:#666">Conseils: {', '.join(z.get('advice',[]))}</span>
        </li>
        """
    
    # Couleurs dominantes
    colors_html = ""
    for color in safe.get("dominant_colors", [])[:5]:
        rgb = color.get("color", [128,128,128])
        colors_html += f"""
        <div style="display:inline-block; margin:5px;">
            <div style="width:60px; height:60px; background:rgb({rgb[0]},{rgb[1]},{rgb[2]}); 
                        border:2px solid #ddd; border-radius:5px;"></div>
            <div style="text-align:center; font-size:0.8em; color:#666;">RGB({rgb[0]},{rgb[1]},{rgb[2]})</div>
        </div>
        """
    
    # Scène
    scene = safe.get("scene") or {}
    scene_html = f"""
    <div style="background:#f0f8ff; padding:15px; border-radius:8px; margin:20px 0;">
        <h3 style="margin:0 0 10px 0; color:#2196F3;">🌍 Scène Détectée</h3>
        <p style="margin:0; font-size:1.2em;"><b>{scene.get('scene_type', 'N/A')}</b> 
           (confiance: {scene.get('probability', 0):.1%})</p>
    </div>
    """
    
    # Légende
    caption_html = f"""
    <div style="background:#fff3e0; padding:15px; border-radius:8px; margin:20px 0;">
        <h3 style="margin:0 0 10px 0; color:#FF9800;">💬 Légende IA</h3>
        <p style="margin:0; font-style:italic;">"{safe.get('caption', 'N/A')}"</p>
        <p style="margin:5px 0 0 0; font-size:0.85em; color:#666;">Source: {safe.get('caption_source', 'N/A')}</p>
    </div>
    """
    
    # Styles
    style_affinities = safe.get("v2", {}).get("style_affinities", {})
    best_match = style_affinities.get("best_match")
    top_5 = style_affinities.get("top_5", [])
    
    best_match_html = ""
    if best_match:
        score_percent = int(best_match['score'] * 100)
        characteristics = "<li>" + "</li><li>".join(best_match.get('key_characteristics', [])) + "</li>"
        
        best_match_html = f"""
        <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; padding:20px; border-radius:10px; margin:20px 0;">
            <h3 style="margin:0 0 10px 0; color:white;">🏆 Meilleur Style : {best_match['label']}</h3>
            <div style="font-size:2em; font-weight:bold; margin:10px 0;">{score_percent}%</div>
            <p style="opacity:0.9; margin:10px 0;">{best_match.get('description', '')}</p>
            <details style="margin-top:10px; background:rgba(255,255,255,0.1); padding:10px; border-radius:5px;">
                <summary style="cursor:pointer; font-weight:bold;">Détails du style</summary>
                <ul style="margin:10px 0;">{characteristics}</ul>
            </details>
        </div>
        """
    
    top5_html = ""
    if top_5:
        top5_html = '<table style="width:100%; border-collapse:collapse; margin:20px 0;"><tr><th>Style</th><th>Score</th><th>Catégorie</th></tr>'
        for style in top_5:
            score_percent = int(style['score'] * 100)
            bar_color = "#4CAF50" if score_percent >= 70 else "#FF9800" if score_percent >= 50 else "#f44336"
            top5_html += f"""
            <tr>
                <td><b>{style['label']}</b></td>
                <td style="width:150px;">
                    <div style="background:#f0f0f0; border-radius:10px; height:25px; overflow:hidden;">
                        <div style="background:{bar_color}; width:{score_percent}%; height:100%; 
                                    display:flex; align-items:center; justify-content:center; 
                                    color:white; font-weight:bold;">{score_percent}%</div>
                    </div>
                </td>
                <td>{style.get('category', 'N/A')}</td>
            </tr>
            """
        top5_html += '</table>'
    
    # Sections
    expert_advice_html = build_expert_advice_html(analysis)
    composition_html = build_composition_rules_html(analysis)
    
    # HTML COMPLET
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Rapport d'Analyse Photo - {Path(analysis.get('path', 'photo')).name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-left: 4px solid #2196F3;
                padding-left: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background: #f9f9f9;
            }}
            tr:hover {{
                background: #f1f1f1;
            }}
            pre {{
                background: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                max-height: 400px;
            }}
            details {{
                margin: 10px 0;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 5px;
            }}
            summary {{
                font-weight: bold;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📸 Rapport d'Analyse Photo Professionnelle</h1>
            
            {scene_html}
            {caption_html}
            
            {expert_advice_html}
            
            {composition_html}
            
            {best_match_html}
            
            <h2>🎨 Top 5 Styles Compatibles</h2>
            {top5_html}
            
            <h2>🎨 Couleurs Dominantes</h2>
            <div style="margin:20px 0;">{colors_html}</div>
            
            <h2>📊 Résumé Technique</h2>
            <table>
                <tr><th>Critère</th><th>Évaluation</th><th>Conseil</th></tr>
                {''.join([f"<tr><td>{c}</td><td>{e}</td><td>{co}</td></tr>" for c, e, co in rows])}
            </table>
            
            <h2>🗺️ Analyse par Zones</h2>
            <ul>{zones_html if zones_html else '<li>Aucune zone détectée</li>'}</ul>
            
            <h2>📷 Données EXIF</h2>
            <ul>{exif_html if exif_html else '<li>Aucune donnée EXIF</li>'}</ul>
            
            <details>
                <summary>🔍 JSON Complet de l'Analyse</summary>
                <pre>{json.dumps(safe, indent=2, ensure_ascii=False)}</pre>
            </details>
        </div>
    </body>
    </html>
    """
    
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"📄 Rapport HTML généré : {out}")
    return out


def annotate_image(image_path, analysis, out=None):
    """Génération d'image annotée"""
    if out is None:
        image_name = Path(image_path).stem
        out = f"annotated_{image_name}.jpg"
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Impossible de charger l'image")
        return None
    
    h, w = img.shape[:2]
    
    # Lignes de tiers
    cv2.line(img, (w//3, 0), (w//3, h), (255, 255, 255), 1)
    cv2.line(img, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 1)
    cv2.line(img, (0, h//3), (w, h//3), (255, 255, 255), 1)
    cv2.line(img, (0, 2*h//3), (w, 2*h//3), (255, 255, 255), 1)
    
    # Objets
    for subj in analysis.get("subjects", []):
        x1, y1, x2, y2 = map(int, subj["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{subj['class']}:{subj['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1-6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Visages
    for face in analysis.get("faces", []):
        x, y, fw, fh = map(int, face["bbox"])
        cv2.rectangle(img, (x, y), (x+fw, y+fh), (0, 128, 255), 2)
    
    # Annotations
    scene_text = f"Scene: {analysis.get('scene', {}).get('scene_type', 'unknown')}"
    cv2.putText(img, scene_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    
    best_style = analysis.get("v2", {}).get("style_affinities", {}).get("best_match")
    if best_style:
        style_text = f"Style: {best_style['label']} ({int(best_style['score']*100)}%)"
        cv2.putText(img, style_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
    
    cv2.imwrite(out, img)
    print(f"🖼️  Image annotée : {out}")
    return out