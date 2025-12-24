# -*- coding: utf-8 -*-
"""
intelligent_advisor.py
Système de conseil intelligent basé sur l'analyse complète
"""

from typing import Dict, List
from photography_knowledge import (
    PhotographyKnowledge, 
    detect_light_condition, 
    detect_genre
)


class IntelligentAdvisor:
    """
    Conseiller photographique intelligent qui analyse le contexte global
    et fournit des recommandations personnalisées
    """
    
    def __init__(self):
        self.knowledge = PhotographyKnowledge()
    
    def analyze_and_advise(self, analysis: Dict, user_level: str = "intermédiaire") -> Dict:
        """
        Analyse complète et génération de conseils personnalisés
        
        Args:
            analysis: Résultat de l'analyse complète de l'image
            user_level: "débutant", "intermédiaire", "avancé"
        
        Returns:
            Dict avec conseils hiérarchisés
        """
        
        # Détection automatique du contexte
        genre = detect_genre(analysis)
        light_condition = detect_light_condition(analysis)
        
        # Récupération des métriques
        brightness = analysis.get('brightness', 128)
        contrast = analysis.get('contrast', 50)
        sharpness = analysis.get('sharpness', 100)
        noise = analysis.get('noise', 20)
        
        # Analyse des défauts
        has_motion_blur = analysis.get('motion_blur_detected', False)
        has_vignette = analysis.get('vignette', False)
        has_chrom_ab = analysis.get('chrom_ab', (False, 0))[0]
        horizon_angle = abs(analysis.get('horizon_angle', 0))
        
        # Analyse composition
        comp_eligibility = analysis.get('composition_eligibility', {})
        comp_score = comp_eligibility.get('global_score', 0.5)
        
        # Analyse style
        best_style = analysis.get('v2', {}).get('style_affinities', {}).get('best_match')
        
        # ========================= GÉNÉRATION DES CONSEILS =========================
        
        advice = {
            "context": {
                "genre": genre,
                "light_condition": light_condition,
                "user_level": user_level
            },
            "immediate_fixes": [],      # Corrections immédiates (post-prod)
            "shooting_improvements": [], # Améliorations pour la prochaine fois
            "technical_tips": [],        # Conseils techniques
            "creative_suggestions": [],  # Opportunités créatives
            "learning_resources": []     # Pour progresser
        }
        
        # ========================= CORRECTIONS IMMÉDIATES =========================
        
        if horizon_angle > 3:
            advice["immediate_fixes"].append({
                "priority": "high",
                "issue": f"Horizon penché de {horizon_angle:.1f}°",
                "fix": "Outil de redressement : rotation de {:.1f}°".format(-horizon_angle),
                "tool": "Lightroom: Crop Tool > Angle slider"
            })
        
        if brightness < 90:
            ev_needed = (120 - brightness) / 50
            advice["immediate_fixes"].append({
                "priority": "high",
                "issue": "Photo sous-exposée",
                "fix": f"Augmenter exposition de +{ev_needed:.1f} EV",
                "tool": "Lightroom: Exposure +{:.1f}".format(ev_needed)
            })
        
        if brightness > 180:
            ev_needed = (brightness - 150) / 50
            advice["immediate_fixes"].append({
                "priority": "high",
                "issue": "Photo surexposée",
                "fix": f"Réduire exposition de -{ev_needed:.1f} EV",
                "tool": "Lightroom: Exposure -{:.1f}, Highlights -50".format(ev_needed)
            })
        
        if has_vignette:
            advice["immediate_fixes"].append({
                "priority": "medium",
                "issue": "Vignetage détecté",
                "fix": "Activer la correction du profil objectif",
                "tool": "Lightroom: Lens Corrections > Enable Profile"
            })
        
        if has_chrom_ab:
            advice["immediate_fixes"].append({
                "priority": "medium",
                "issue": "Aberration chromatique présente",
                "fix": "Activer la suppression des franges colorées",
                "tool": "Lightroom: Lens Corrections > Remove Chromatic Aberration"
            })
        
        if comp_score < 0.6:
            advice["immediate_fixes"].append({
                "priority": "high",
                "issue": "Composition faible (règles non respectées)",
                "fix": "Recadrer selon règle des tiers ou nombre d'or",
                "tool": "Lightroom: Crop Tool > Overlay > Rule of Thirds"
            })
        
        # ========================= AMÉLIORATIONS SHOOTING =========================
        
        if has_motion_blur:
            advice["shooting_improvements"].append({
                "priority": "critical",
                "issue": "Flou de mouvement détecté",
                "why": "Vitesse d'obturation trop lente pour figer le sujet",
                "solution": "Augmenter vitesse : minimum 1/{:.0f}s recommandé".format(max(250, 1 / (sharpness / 100))),
                "alternative": "Ou stabiliser : trépied / monopode / appui"
            })
        
        if sharpness < 100:
            advice["shooting_improvements"].append({
                "priority": "high",
                "issue": "Netteté insuffisante",
                "causes": ["Vitesse trop lente", "Autofocus raté", "Objectif de mauvaise qualité"],
                "solutions": [
                    "Vitesse minimum : 1/125s (ou 1/[focale] × 2)",
                    "Mode AF : Single point AF sur le sujet principal",
                    "Si faible lumière : augmenter ISO plutôt que ralentir vitesse"
                ]
            })
        
        if noise > 40:
            advice["shooting_improvements"].append({
                "priority": "medium",
                "issue": f"Bruit numérique élevé ({noise:.0f})",
                "causes": ["ISO trop élevé", "Sous-exposition puis récupération"],
                "solutions": [
                    "Baisser ISO : privilégier 100-800 si possible",
                    "Augmenter exposition à la prise de vue",
                    "Utiliser un trépied pour ISO plus bas"
                ]
            })
        
        # Conseils spécifiques au genre
        contextual = self.knowledge.get_contextual_advice(genre, light_condition, "unknown", user_level)
        
        for tip in contextual.get("genre_tips", [])[:3]:
            advice["technical_tips"].append({
                "category": f"{genre.title()} Photography",
                "tip": tip
            })
        
        for tip in contextual.get("light_tips", [])[:3]:
            advice["shooting_improvements"].append({
                "priority": "medium",
                "issue": f"Lumière : {light_condition.replace('_', ' ').title()}",
                "solution": tip
            })
        
        # ========================= SUGGESTIONS CRÉATIVES =========================
        
        if best_style and best_style.get('score', 0) > 0.6:
            style_tips = best_style.get('pro_tips', [])
            for tip in style_tips[:2]:
                advice["creative_suggestions"].append({
                    "style": best_style.get('label'),
                    "suggestion": tip
                })
        
        # Opportunités selon contexte
        if genre == "portrait" and brightness > 150:
            advice["creative_suggestions"].append({
                "opportunity": "High Key Portrait",
                "suggestion": "Votre lumière est déjà élevée, poussez vers +2 EV pour un effet High Key aérien"
            })
        
        if genre == "landscape" and light_condition == "golden_hour":
            advice["creative_suggestions"].append({
                "opportunity": "Golden Hour Magic",
                "suggestion": "Lumière parfaite ! Photographiez en série et faites un bracketing pour HDR"
            })
        
        # ========================= RESSOURCES D'APPRENTISSAGE =========================
        
        if user_level == "débutant":
            advice["learning_resources"] = [
                {
                    "topic": "Exposition (Triangle : ISO, Ouverture, Vitesse)",
                    "why": "Base fondamentale de la photographie",
                    "priority": "essential"
                },
                {
                    "topic": "Composition (Règle des tiers, Lignes directrices)",
                    "why": "Transforme une photo technique en œuvre visuelle",
                    "priority": "essential"
                },
                {
                    "topic": "Lumière (Direction, Qualité, Température)",
                    "why": "La lumière EST la photographie",
                    "priority": "essential"
                }
            ]
        elif user_level == "intermédiaire":
            advice["learning_resources"] = [
                {
                    "topic": "Post-traitement avancé (Courbes, HSL, Masques)",
                    "why": "Révéler le potentiel de vos RAW",
                    "priority": "important"
                },
                {
                    "topic": "Lumière artificielle (Flash, Modificateurs)",
                    "why": "Contrôle total de la lumière",
                    "priority": "important"
                },
                {
                    "topic": "Développer votre style",
                    "why": "Se démarquer et affirmer sa vision",
                    "priority": "recommended"
                }
            ]
        else:  # avancé
            advice["learning_resources"] = [
                {
                    "topic": "Vision artistique et narration",
                    "why": "Au-delà de la technique, raconter une histoire",
                    "priority": "essential"
                },
                {
                    "topic": "Projet photo long terme",
                    "why": "Cohérence et profondeur",
                    "priority": "important"
                }
            ]
        
        # ========================= PRIORISATION ET SCORING =========================
        
        advice["summary"] = {
            "critical_issues": len([f for f in advice["immediate_fixes"] if f.get("priority") == "high"]),
            "can_be_saved": comp_score > 0.4 or len(advice["immediate_fixes"]) <= 3,
            "overall_assessment": self._generate_assessment(analysis, advice),
            "next_steps": self._prioritize_next_steps(advice)
        }
        
        return advice
    
    def _generate_assessment(self, analysis: Dict, advice: Dict) -> str:
        """Génère une évaluation globale en langage naturel"""
        
        quality = analysis.get('quality_score', 50)
        comp_score = analysis.get('composition_eligibility', {}).get('global_score', 0.5)
        critical_issues = advice["summary"]["critical_issues"]
        
        if quality >= 80 and comp_score >= 0.8:
            return "📸 Photo excellente ! Quelques ajustements mineurs suffiront. Vous maîtrisez bien votre sujet."
        
        elif quality >= 70 and comp_score >= 0.6:
            return "✅ Bonne base technique. La composition peut être améliorée facilement. Photo exploitable."
        
        elif quality >= 60 or comp_score >= 0.5:
            return "⚠️ Photo correcte mais nécessite du travail. Les corrections sont réalisables en post-production."
        
        elif critical_issues >= 3:
            return "🔧 Plusieurs problèmes techniques majeurs. Considérez reprendre la photo avec les conseils fournis."
        
        else:
            return "📚 Photo d'apprentissage. Étudiez les conseils pour votre prochaine session."
    
    def _prioritize_next_steps(self, advice: Dict) -> List[str]:
        """Priorise les 3 prochaines actions à faire"""
        
        steps = []
        
        # Corrections immédiates critiques
        critical_fixes = [f for f in advice["immediate_fixes"] if f.get("priority") == "high"]
        if critical_fixes:
            steps.append(f"1. CORRECTION : {critical_fixes[0]['issue']} → {critical_fixes[0]['fix']}")
        
        # Amélioration shooting la plus importante
        critical_shooting = [s for s in advice["shooting_improvements"] if s.get("priority") in ["critical", "high"]]
        if critical_shooting and len(steps) < 3:
            steps.append(f"{len(steps)+1}. PROCHAINE FOIS : {critical_shooting[0]['solution']}")
        
        # Apprentissage essentiel
        essential_learning = [r for r in advice.get("learning_resources", []) if r.get("priority") == "essential"]
        if essential_learning and len(steps) < 3:
            steps.append(f"{len(steps)+1}. APPRENDRE : {essential_learning[0]['topic']}")
        
        return steps[:3]


def generate_ai_coach_report(analysis: Dict, user_level: str = "intermédiaire") -> str:
    """
    Génère un rapport textuel comme si un photographe pro analysait l'image
    """
    
    advisor = IntelligentAdvisor()
    advice = advisor.analyze_and_advise(analysis, user_level)
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           🎓 ANALYSE PAR UN PHOTOGRAPHE EXPERT              ║
╚══════════════════════════════════════════════════════════════╝

📋 CONTEXTE DÉTECTÉ
Genre : {advice['context']['genre'].title()}
Lumière : {advice['context']['light_condition'].replace('_', ' ').title()}
Niveau : {advice['context']['user_level'].title()}

{advice['summary']['overall_assessment']}

"""
    
    if advice['immediate_fixes']:
        report += "\n🔧 CORRECTIONS IMMÉDIATES (Post-production)\n" + "─" * 60 + "\n"
        for i, fix in enumerate(advice['immediate_fixes'][:5], 1):
            priority_icon = "🔴" if fix['priority'] == "high" else "🟠"
            report += f"{priority_icon} {i}. {fix['issue']}\n"
            report += f"   → {fix['fix']}\n"
            report += f"   💡 {fix['tool']}\n\n"
    
    if advice['shooting_improvements']:
        report += "\n📸 POUR VOS PROCHAINES PHOTOS\n" + "─" * 60 + "\n"
        for i, imp in enumerate(advice['shooting_improvements'][:5], 1):
            report += f"{i}. {imp.get('issue', imp.get('solution', ''))}\n"
            if 'solution' in imp and 'issue' in imp:
                report += f"   → {imp['solution']}\n"
            if 'alternative' in imp:
                report += f"   ℹ️  Alternative : {imp['alternative']}\n"
            report += "\n"
    
    if advice['creative_suggestions']:
        report += "\n💡 OPPORTUNITÉS CRÉATIVES\n" + "─" * 60 + "\n"
        for sugg in advice['creative_suggestions'][:3]:
            if 'style' in sugg:
                report += f"Style {sugg['style']} : {sugg['suggestion']}\n"
            else:
                report += f"{sugg.get('opportunity', 'Suggestion')} : {sugg['suggestion']}\n"
            report += "\n"
    
    if advice['summary']['next_steps']:
        report += "\n⚡ VOS 3 PROCHAINES ACTIONS\n" + "─" * 60 + "\n"
        for step in advice['summary']['next_steps']:
            report += f"  {step}\n"
    
    return report