# -*- coding: utf-8 -*-
"""
composition_rules.py
Analyse des règles de composition photographique classiques
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class CompositionRule:
    """Classe de base pour une règle de composition"""
    
    def __init__(self, name: str, category: str, importance: str):
        self.name = name
        self.category = category  # "placement", "balance", "depth", "leading_lines", etc.
        self.importance = importance  # "essential", "important", "recommended"
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        """
        Évalue si la règle est respectée
        
        Returns:
            {
                'respected': bool,
                'score': float (0-1),
                'message': str,
                'suggestions': List[str],
                'adjustments': List[Dict]
            }
        """
        raise NotImplementedError


class RuleOfThirds(CompositionRule):
    """Règle des tiers"""
    
    def __init__(self):
        super().__init__("Règle des tiers", "placement", "essential")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        h, w = img_rgb.shape[:2]
        
        # Points forts (intersections des tiers)
        strong_points = [
            (w/3, h/3), (2*w/3, h/3),
            (w/3, 2*h/3), (2*w/3, 2*h/3)
        ]
        
        if not subjects:
            return {
                'respected': False,
                'score': 0.0,
                'message': "Aucun sujet détecté pour évaluer le placement",
                'suggestions': [],
                'adjustments': []
            }
        
        # Sujet principal
        main_subject = max(subjects, key=lambda s: (s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]))
        x1, y1, x2, y2 = main_subject["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Distance au point fort le plus proche
        min_dist = min(np.sqrt((cx - px)**2 + (cy - py)**2) for px, py in strong_points)
        normalized_dist = min_dist / np.sqrt(w**2 + h**2)
        
        # Score : plus proche = mieux
        score = max(0, 1 - normalized_dist * 8)  # Distance normalisée
        
        respected = score > 0.7
        
        suggestions = []
        adjustments = []
        
        if not respected:
            # Trouver le meilleur point fort
            best_point = min(strong_points, key=lambda p: np.sqrt((cx - p[0])**2 + (cy - p[1])**2))
            offset_x = best_point[0] - cx
            offset_y = best_point[1] - cy
            
            suggestions.append(f"Recadrer pour placer le sujet sur un point fort")
            suggestions.append(f"Déplacer de {int(offset_x)}px horizontalement, {int(offset_y)}px verticalement")
            
            adjustments.append({
                'type': 'crop',
                'action': 'recenter_subject',
                'offset_x': int(offset_x),
                'offset_y': int(offset_y),
                'difficulty': 'easy'
            })
        else:
            suggestions.append("✅ Sujet bien placé sur un point fort")
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': f"Sujet à {int(min_dist)}px du point fort le plus proche",
            'suggestions': suggestions,
            'adjustments': adjustments
        }


class GoldenRatio(CompositionRule):
    """Nombre d'or / Spirale de Fibonacci"""
    
    def __init__(self):
        super().__init__("Nombre d'or", "placement", "recommended")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        h, w = img_rgb.shape[:2]
        phi = 1.618
        
        # Points forts du nombre d'or
        golden_points = [
            (w/phi, h/phi), (w - w/phi, h/phi),
            (w/phi, h - h/phi), (w - w/phi, h - h/phi)
        ]
        
        if not subjects:
            return {
                'respected': False,
                'score': 0.0,
                'message': "Aucun sujet pour évaluer le nombre d'or",
                'suggestions': [],
                'adjustments': []
            }
        
        main_subject = max(subjects, key=lambda s: (s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]))
        x1, y1, x2, y2 = main_subject["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        min_dist = min(np.sqrt((cx - px)**2 + (cy - py)**2) for px, py in golden_points)
        normalized_dist = min_dist / np.sqrt(w**2 + h**2)
        score = max(0, 1 - normalized_dist * 8)
        
        respected = score > 0.75
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': f"Alignement au nombre d'or : {int(score*100)}%",
            'suggestions': ["Composition harmonieuse selon le nombre d'or"] if respected else ["Ajuster le cadrage selon la spirale de Fibonacci"],
            'adjustments': []
        }


class LeadingLines(CompositionRule):
    """Lignes directrices"""
    
    def __init__(self):
        super().__init__("Lignes directrices", "leading_lines", "important")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Détection de lignes avec Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=int(min(img_rgb.shape[:2])/4), 
                                maxLineGap=20)
        
        if lines is None or len(lines) < 3:
            return {
                'respected': False,
                'score': 0.2,
                'message': f"Peu de lignes directrices détectées ({len(lines) if lines is not None else 0})",
                'suggestions': ["Chercher des éléments linéaires pour guider l'œil", "Routes, rivières, ponts, rampes créent des lignes"],
                'adjustments': []
            }
        
        # Analyser direction des lignes vers le sujet
        score = min(1.0, len(lines) / 10)  # Plus de lignes = mieux
        respected = len(lines) >= 5
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': f"{len(lines)} lignes directrices détectées",
            'suggestions': ["✅ Bonnes lignes directrices présentes"] if respected else ["Renforcer les lignes avec recadrage ou perspective"],
            'adjustments': []
        }


class Symmetry(CompositionRule):
    """Symétrie"""
    
    def __init__(self):
        super().__init__("Symétrie", "balance", "recommended")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        h, w = img_rgb.shape[:2]
        
        # Symétrie verticale
        left_half = img_rgb[:, :w//2]
        right_half = cv2.flip(img_rgb[:, w//2:], 1)
        
        # Redimensionner pour comparer
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = cv2.resize(left_half, (min_w, h))
        right_half = cv2.resize(right_half, (min_w, h))
        
        # Différence
        diff = cv2.absdiff(left_half, right_half)
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        respected = symmetry_score > 0.7
        
        return {
            'respected': respected,
            'score': round(symmetry_score, 3),
            'message': f"Symétrie : {int(symmetry_score*100)}%",
            'suggestions': ["✅ Composition symétrique réussie"] if respected else ["Symétrie faible (pas forcément un problème)"],
            'adjustments': []
        }


class BalanceRule(CompositionRule):
    """Équilibre visuel"""
    
    def __init__(self):
        super().__init__("Équilibre visuel", "balance", "essential")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        if not subjects:
            return {
                'respected': True,
                'score': 0.5,
                'message': "Aucun sujet pour évaluer l'équilibre",
                'suggestions': [],
                'adjustments': []
            }
        
        h, w = img_rgb.shape[:2]
        
        # Centre de masse des sujets
        total_area = 0
        weighted_x = 0
        weighted_y = 0
        
        for subj in subjects:
            x1, y1, x2, y2 = subj["bbox"]
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            weighted_x += cx * area
            weighted_y += cy * area
            total_area += area
        
        if total_area > 0:
            center_x = weighted_x / total_area
            center_y = weighted_y / total_area
        else:
            return {'respected': False, 'score': 0.0, 'message': '', 'suggestions': [], 'adjustments': []}
        
        # Distance au centre de l'image
        image_center_x = w / 2
        image_center_y = h / 2
        
        offset_x = abs(center_x - image_center_x) / w
        offset_y = abs(center_y - image_center_y) / h
        
        # Score d'équilibre (proche du centre ou équilibré sur les tiers)
        balance_score = 1.0 - (offset_x + offset_y) / 2
        
        respected = balance_score > 0.6
        
        suggestions = []
        if not respected:
            suggestions.append("Rééquilibrer la composition")
            if offset_x > 0.3:
                suggestions.append("Trop de poids visuel à gauche ou droite")
            if offset_y > 0.3:
                suggestions.append("Trop de poids visuel en haut ou bas")
        else:
            suggestions.append("✅ Composition équilibrée")
        
        return {
            'respected': respected,
            'score': round(balance_score, 3),
            'message': f"Centre de masse à ({int(center_x)}, {int(center_y)})",
            'suggestions': suggestions,
            'adjustments': []
        }


class NegativeSpace(CompositionRule):
    """Espace négatif"""
    
    def __init__(self):
        super().__init__("Espace négatif", "depth", "recommended")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        if not subjects:
            return {
                'respected': True,
                'score': 1.0,
                'message': "Beaucoup d'espace négatif (pas de sujet détecté)",
                'suggestions': ["Photo minimaliste"],
                'adjustments': []
            }
        
        h, w = img_rgb.shape[:2]
        total_pixels = h * w
        
        # Surface occupée par les sujets
        subject_area = sum((s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]) for s in subjects)
        subject_percent = subject_area / total_pixels
        
        # Espace négatif
        negative_space = 1.0 - subject_percent
        
        # Bon équilibre : 60-80% d'espace négatif
        if 0.6 <= negative_space <= 0.8:
            score = 1.0
            respected = True
            message = f"✅ Excellent équilibre : {int(negative_space*100)}% d'espace négatif"
        elif negative_space > 0.8:
            score = 0.7
            respected = True
            message = f"Beaucoup d'espace négatif ({int(negative_space*100)}%), style minimaliste"
        else:
            score = 0.5
            respected = False
            message = f"Peu d'espace négatif ({int(negative_space*100)}%), composition chargée"
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': message,
            'suggestions': ["Composition aérée"] if respected else ["Recadrer plus serré ou éloigner le sujet"],
            'adjustments': []
        }


class FramingRule(CompositionRule):
    """Cadrage naturel"""
    
    def __init__(self):
        super().__init__("Cadrage naturel", "depth", "recommended")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        # Détecter si les bords de l'image ont des éléments sombres (cadrage naturel)
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Bords (10% de chaque côté)
        border_size = int(min(h, w) * 0.1)
        
        top_border = gray[:border_size, :]
        bottom_border = gray[-border_size:, :]
        left_border = gray[:, :border_size]
        right_border = gray[:, -border_size:]
        
        borders = [top_border, bottom_border, left_border, right_border]
        border_brightness = [np.mean(b) for b in borders]
        
        # Centre de l'image
        center_region = gray[border_size:-border_size, border_size:-border_size]
        center_brightness = np.mean(center_region)
        
        # Cadrage naturel = bords plus sombres que le centre
        dark_borders = sum(1 for b in border_brightness if b < center_brightness - 30)
        
        score = dark_borders / 4.0
        respected = score >= 0.5
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': f"{dark_borders}/4 bords créent un cadrage naturel",
            'suggestions': ["✅ Cadrage naturel présent"] if respected else ["Utiliser portes, fenêtres, branches pour cadrer le sujet"],
            'adjustments': []
        }


class DepthOfField(CompositionRule):
    """Profondeur de champ"""
    
    def __init__(self):
        super().__init__("Profondeur de champ", "depth", "important")
    
    def evaluate(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        sharpness = analysis.get('sharpness', 0)
        
        # Calculer variation de netteté dans l'image (proxy pour PDC)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Découper en zones
        zones = [
            gray[:h//3, :],          # Haut
            gray[h//3:2*h//3, :],    # Centre
            gray[2*h//3:, :]         # Bas
        ]
        
        zone_sharpness = [cv2.Laplacian(z, cv2.CV_64F).var() for z in zones]
        variation = np.std(zone_sharpness)
        
        # Grande variation = bonne PDC (sujet net, fond flou)
        score = min(1.0, variation / 100)
        respected = score > 0.5
        
        return {
            'respected': respected,
            'score': round(score, 3),
            'message': f"Variation de netteté : {int(variation)}",
            'suggestions': ["✅ Bonne séparation sujet/arrière-plan"] if respected else ["Utiliser une plus grande ouverture (f/2.8, f/1.8) pour flouter le fond"],
            'adjustments': []
        }


class CompositionAnalyzer:
    """Analyseur complet des règles de composition"""
    
    def __init__(self):
        self.rules = [
            RuleOfThirds(),
            GoldenRatio(),
            LeadingLines(),
            Symmetry(),
            BalanceRule(),
            NegativeSpace(),
            FramingRule(),
            DepthOfField()
        ]
    
    def analyze_all_rules(self, img_rgb: np.ndarray, subjects: List[Dict], analysis: Dict) -> Dict:
        """Analyse toutes les règles"""
        results = {}
        
        for rule in self.rules:
            try:
                result = rule.evaluate(img_rgb, subjects, analysis)
                results[rule.name] = {
                    'category': rule.category,
                    'importance': rule.importance,
                    **result
                }
            except Exception as e:
                print(f"⚠️ Erreur règle {rule.name}: {e}")
                results[rule.name] = {
                    'category': rule.category,
                    'importance': rule.importance,
                    'respected': False,
                    'score': 0.0,
                    'message': f"Erreur: {e}",
                    'suggestions': [],
                    'adjustments': []
                }
        
        # Score global
        essential_rules = [r for r in results.values() if r['importance'] == 'essential']
        important_rules = [r for r in results.values() if r['importance'] == 'important']
        
        essential_score = np.mean([r['score'] for r in essential_rules]) if essential_rules else 0.5
        important_score = np.mean([r['score'] for r in important_rules]) if important_rules else 0.5
        
        global_score = essential_score * 0.6 + important_score * 0.4
        
        # Compte des règles respectées
        respected_count = sum(1 for r in results.values() if r['respected'])
        total_count = len(results)
        
        return {
            'rules': results,
            'summary': {
                'global_score': round(global_score, 3),
                'global_percentage': int(global_score * 100),
                'respected_count': respected_count,
                'total_count': total_count,
                'respect_rate': f"{respected_count}/{total_count}",
                'grade': self._get_grade(global_score)
            },
            'prioritized_suggestions': self._prioritize_suggestions(results)
        }
    
    def _get_grade(self, score: float) -> str:
        """Attribution d'une note"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Très bon)"
        elif score >= 0.7:
            return "B (Bon)"
        elif score >= 0.6:
            return "C (Correct)"
        elif score >= 0.5:
            return "D (Améliorable)"
        else:
            return "E (À retravailler)"
    
    def _prioritize_suggestions(self, results: Dict) -> List[str]:
        """Priorise les suggestions selon l'importance"""
        essential = []
        important = []
        recommended = []
        
        for rule_name, rule_data in results.items():
            if not rule_data['respected'] and rule_data['suggestions']:
                suggestion = f"{rule_name}: {rule_data['suggestions'][0]}"
                
                if rule_data['importance'] == 'essential':
                    essential.append(suggestion)
                elif rule_data['importance'] == 'important':
                    important.append(suggestion)
                else:
                    recommended.append(suggestion)
        
        return essential + important + recommended
    
    def get_eligibility_verdict(self, analysis_result: Dict) -> Dict:
        """
        Détermine si la photo est éligible pour les règles de composition classiques
        et quels ajustements sont nécessaires
        """
        summary = analysis_result['summary']
        rules = analysis_result['rules']
        
        global_score = summary['global_score']
        
        # Critères d'éligibilité
        essential_rules = {name: data for name, data in rules.items() if data['importance'] == 'essential'}
        essential_respected = all(r['respected'] for r in essential_rules.values())
        
        # Verdict
        if global_score >= 0.8:
            verdict = "EXCELLENT"
            eligibility = "eligible_perfect"
            message = "✅ Photo excellente ! Respecte les règles classiques de composition. Prête pour publication."
            color = "#4CAF50"
            bg = "#e8f5e9"
            icon = "✅"
        elif global_score >= 0.7:
            verdict = "BON"
            eligibility = "eligible_minor"
            message = "✅ Photo éligible avec ajustements mineurs. Quelques améliorations simples suffisent."
            color = "#8BC34A"
            bg = "#f1f8e9"
            icon = "✅"
        elif global_score >= 0.6 and essential_respected:
            verdict = "CORRECT"
            eligibility = "eligible_adjustments"
            message = "⚠️ Photo éligible SOUS CONDITION. Ajustements recommandés pour atteindre l'excellence."
            color = "#FF9800"
            bg = "#fff3e0"
            icon = "⚠️"
        elif global_score >= 0.5:
            verdict = "AMÉLIORABLE"
            eligibility = "requires_work"
            message = "⚠️ Photo qui NÉCESSITE du travail. Plusieurs règles importantes non respectées."
            color = "#FF5722"
            bg = "#fbe9e7"
            icon = "⚠️"
        else:
            verdict = "À RETRAVAILLER"
            eligibility = "not_eligible"
            message = "❌ Photo NON ÉLIGIBLE en l'état. Composition à revoir entièrement (recadrage, recomposition)."
            color = "#F44336"
            bg = "#ffebee"
            icon = "❌"
        
        # Ajustements nécessaires par difficulté
        adjustments = {
            'easy': [],      # Recadrage simple
            'medium': [],    # Ajustements locaux
            'hard': [],      # Retouche avancée
            'impossible': [] # Nécessite reprise photo
        }
        
        for rule_name, rule_data in rules.items():
            if not rule_data['respected'] and rule_data['adjustments']:
                for adj in rule_data['adjustments']:
                    difficulty = adj.get('difficulty', 'medium')
                    adjustments[difficulty].append({
                        'rule': rule_name,
                        'action': adj.get('action'),
                        'details': adj
                    })
        
        return {
            'verdict': verdict,
            'eligibility': eligibility,
            'message': message,
            'color': color,
            'bg_color': bg,
            'icon': icon,
            'global_score': global_score,
            'can_be_fixed': eligibility not in ['not_eligible'],
            'adjustments_needed': adjustments,
            'estimated_work_time': self._estimate_work_time(adjustments),
            'priority_fixes': analysis_result['prioritized_suggestions'][:3]
        }
    
    def _estimate_work_time(self, adjustments: Dict) -> str:
        """Estime le temps de travail nécessaire"""
        easy_count = len(adjustments['easy'])
        medium_count = len(adjustments['medium'])
        hard_count = len(adjustments['hard'])
        impossible_count = len(adjustments['impossible'])
        
        if impossible_count > 0:
            return "Impossible sans reprendre la photo"
        
        total_minutes = easy_count * 2 + medium_count * 5 + hard_count * 15
        
        if total_minutes == 0:
            return "Aucun travail nécessaire"
        elif total_minutes <= 5:
            return "< 5 minutes (rapide)"
        elif total_minutes <= 15:
            return "10-15 minutes (modéré)"
        elif total_minutes <= 30:
            return "15-30 minutes (conséquent)"
        else:
            return f"~{total_minutes} minutes (important)"