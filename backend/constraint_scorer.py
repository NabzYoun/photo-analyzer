# -*- coding: utf-8 -*-
"""
constraint_scorer.py
Système de scoring avancé avec pondération intelligente des contraintes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ConstraintScorer:
    """
    Système de scoring qui calcule un score de compatibilité 
    basé sur des contraintes strictes et souples
    """
    
    def __init__(self):
        self.penalty_weights = {
            'critical': 1.0,    # Éliminatoire
            'high': 0.3,        # Pénalité forte
            'medium': 0.15,     # Pénalité moyenne
            'low': 0.05         # Pénalité faible
        }
    
    def evaluate_constraint(
        self, 
        constraint_type: str,
        current_value: float,
        target_value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        priority: str = 'medium'
    ) -> Tuple[bool, float, str]:
        """
        Évalue une contrainte individuelle
        
        Returns:
            (passed, penalty_score, message)
        """
        
        # Contraintes strictes (éliminatoires)
        if priority == 'critical':
            if min_value is not None and current_value < min_value:
                return (False, 1.0, f"{constraint_type}: {current_value:.1f} < {min_value} (CRITIQUE)")
            if max_value is not None and current_value > max_value:
                return (False, 1.0, f"{constraint_type}: {current_value:.1f} > {max_value} (CRITIQUE)")
        
        # Contraintes souples (pénalités graduelles)
        penalty = 0.0
        message = f"{constraint_type}: OK"
        
        if target_value is not None:
            # Distance normalisée à la cible
            if min_value is not None and max_value is not None:
                range_size = max_value - min_value
                if range_size > 0:
                    distance = abs(current_value - target_value) / range_size
                    penalty = distance * self.penalty_weights.get(priority, 0.1)
                    message = f"{constraint_type}: {current_value:.1f} vs {target_value:.1f} (écart: {distance:.2f})"
        
        elif min_value is not None and current_value < min_value:
            # En dessous du minimum (mais pas critique)
            distance = (min_value - current_value) / min_value if min_value > 0 else 0
            penalty = min(1.0, distance) * self.penalty_weights.get(priority, 0.1)
            message = f"{constraint_type}: {current_value:.1f} < {min_value} (pénalité: {penalty:.2f})"
        
        elif max_value is not None and current_value > max_value:
            # Au-dessus du maximum (mais pas critique)
            distance = (current_value - max_value) / max_value if max_value > 0 else 0
            penalty = min(1.0, distance) * self.penalty_weights.get(priority, 0.1)
            message = f"{constraint_type}: {current_value:.1f} > {max_value} (pénalité: {penalty:.2f})"
        
        return (True, penalty, message)
    
    def evaluate_all_constraints(
        self,
        metrics: Dict,
        constraints: Dict
    ) -> Tuple[bool, float, List[str]]:
        """
        Évalue toutes les contraintes d'un profil
        
        Returns:
            (all_passed, total_penalty, messages)
        """
        
        all_passed = True
        total_penalty = 0.0
        messages = []
        
        # Contraintes numériques
        numeric_constraints = [
            ('min_contrast', 'contrast', 'high'),
            ('max_contrast', 'contrast', 'medium'),
            ('min_brightness', 'brightness', 'high'),
            ('max_brightness', 'brightness', 'medium'),
            ('min_saturation', 'saturation', 'high'),
            ('max_saturation', 'saturation', 'medium'),
            ('min_sharpness', 'sharpness', 'high'),
            ('max_sharpness', 'sharpness', 'low'),
            ('min_texture', 'texture_score', 'medium'),
        ]
        
        for constraint_key, metric_key, priority in numeric_constraints:
            if constraint_key in constraints:
                threshold = constraints[constraint_key]
                value = metrics.get(metric_key, 0)
                
                is_min = constraint_key.startswith('min_')
                
                passed, penalty, msg = self.evaluate_constraint(
                    constraint_type=metric_key,
                    current_value=value,
                    min_value=threshold if is_min else None,
                    max_value=threshold if not is_min else None,
                    priority=priority
                )
                
                if not passed:
                    all_passed = False
                
                total_penalty += penalty
                messages.append(msg)
        
        return (all_passed, total_penalty, messages)
    
    def calculate_weighted_score(
        self,
        base_score: float,
        constraint_penalty: float,
        constraint_weight: float = 0.3
    ) -> float:
        """
        Combine le score de base avec les pénalités de contraintes
        
        Args:
            base_score: Score calculé via les weights (0-1)
            constraint_penalty: Pénalité totale des contraintes (0-1)
            constraint_weight: Importance des contraintes (0-1)
        
        Returns:
            Score final pondéré (0-1)
        """
        
        # Le score final est une combinaison pondérée
        # Plus le constraint_weight est élevé, plus les contraintes sont importantes
        
        constraint_score = 1.0 - min(1.0, constraint_penalty)
        
        final_score = (
            base_score * (1.0 - constraint_weight) +
            constraint_score * constraint_weight
        )
        
        return np.clip(final_score, 0, 1)


class StyleCompatibilityAnalyzer:
    """
    Analyseur de compatibilité avancé qui fournit des explications détaillées
    """
    
    def __init__(self):
        self.scorer = ConstraintScorer()
    
    def analyze_compatibility(
        self,
        metrics: Dict,
        profile: Dict
    ) -> Dict:
        """
        Analyse complète de la compatibilité avec un profil
        
        Returns:
            {
                'compatible': bool,
                'score': float,
                'base_score': float,
                'constraint_score': float,
                'failed_constraints': List[str],
                'warnings': List[str],
                'recommendations': List[str]
            }
        """
        
        profile_label = profile.get('label', 'Unknown')
        constraints = profile.get('constraints', {})
        
        # Vérifications strictes (éliminatoires)
        failed_strict = []
        
        # N&B vs Couleur
        if constraints.get('black_and_white') and not metrics.get('black_and_white'):
            failed_strict.append("Image en couleur mais profil N&B requis")
        if constraints.get('color') and metrics.get('black_and_white'):
            failed_strict.append("Image N&B mais profil couleur requis")
        
        # Visages
        if constraints.get('requires_face'):
            num_faces = metrics.get('num_faces', 0)
            min_faces = constraints.get('min_faces', 1)
            if num_faces < min_faces:
                failed_strict.append(f"Pas assez de visages ({num_faces} < {min_faces})")
        
        if constraints.get('requires_no_face'):
            num_faces = metrics.get('num_faces', 0)
            if num_faces > 0:
                failed_strict.append(f"Visage présent mais interdit")
        
        # Humains
        if constraints.get('requires_no_human'):
            has_human = metrics.get('has_human', False)
            if has_human:
                failed_strict.append("Présence humaine détectée mais interdite")
        
        if constraints.get('requires_human_presence'):
            has_human = metrics.get('has_human', False)
            if not has_human:
                failed_strict.append("Aucune présence humaine détectée")
        
        # Objets
        if constraints.get('max_objects'):
            num_objects = metrics.get('num_objects', 0)
            max_objects = constraints['max_objects']
            if num_objects > max_objects:
                failed_strict.append(f"Trop d'objets ({num_objects} > {max_objects})")
        
        if constraints.get('min_objects'):
            num_objects = metrics.get('num_objects', 0)
            min_objects = constraints['min_objects']
            if num_objects < min_objects:
                failed_strict.append(f"Pas assez d'objets ({num_objects} < {min_objects})")
        
        # Si contraintes strictes échouent, incompatible
        if failed_strict:
            return {
                'compatible': False,
                'score': 0.0,
                'base_score': 0.0,
                'constraint_score': 0.0,
                'failed_constraints': failed_strict,
                'warnings': [],
                'recommendations': [
                    f"Ce style '{profile_label}' n'est pas adapté à votre image",
                    "Essayez un autre style de la même catégorie"
                ]
            }
        
        # Calcul du score de base (via weights)
        # (Cette partie serait appelée depuis calculate_style_affinity)
        base_score = 0.7  # Exemple
        
        # Évaluation des contraintes souples
        all_passed, penalty, messages = self.scorer.evaluate_all_constraints(
            metrics, constraints
        )
        
        constraint_score = 1.0 - min(1.0, penalty)
        
        # Score final pondéré
        final_score = self.scorer.calculate_weighted_score(
            base_score, penalty, constraint_weight=0.3
        )
        
        # Génération d'avertissements et recommandations
        warnings = []
        recommendations = []
        
        if penalty > 0.5:
            warnings.append(f"Plusieurs contraintes ne sont pas optimales")
            recommendations.append("Ajustez l'exposition et le contraste en post-production")
        
        if constraint_score < 0.7:
            warnings.append("Compatibilité modérée avec ce style")
            recommendations.append("Considérez un style alternatif de la même catégorie")
        
        return {
            'compatible': True,
            'score': round(final_score, 3),
            'base_score': round(base_score, 3),
            'constraint_score': round(constraint_score, 3),
            'failed_constraints': [],
            'warnings': warnings,
            'recommendations': recommendations,
            'constraint_details': messages
        }
    
    def generate_compatibility_report(
        self,
        metrics: Dict,
        profiles: List[Dict],
        top_n: int = 5
    ) -> Dict:
        """
        Génère un rapport complet de compatibilité pour tous les profils
        """
        
        results = []
        
        for profile in profiles:
            analysis = self.analyze_compatibility(metrics, profile)
            analysis['profile_id'] = profile.get('id')
            analysis['profile_label'] = profile.get('label')
            analysis['category'] = profile.get('category')
            results.append(analysis)
        
        # Tri par score décroissant
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Statistiques
        compatible_count = sum(1 for r in results if r['compatible'])
        avg_score = np.mean([r['score'] for r in results if r['compatible']])
        
        return {
            'total_profiles': len(profiles),
            'compatible_profiles': compatible_count,
            'average_score': round(avg_score, 3),
            'top_matches': results[:top_n],
            'all_results': results,
            'summary': f"{compatible_count}/{len(profiles)} profils compatibles (score moyen: {avg_score:.1%})"
        }


# ========================= EXEMPLE D'UTILISATION =========================

def example_usage():
    """Exemple d'utilisation du système de scoring"""
    
    # Métriques d'exemple
    metrics = {
        'brightness': 130,
        'contrast': 75,
        'saturation': 45,
        'sharpness': 180,
        'num_faces': 0,
        'num_objects': 2,
        'has_human': False,
        'black_and_white': True,
        'scene_type': 'mountain'
    }
    
    # Profil d'exemple (Ansel Adams)
    profile = {
        'id': 'ansel_adams',
        'label': 'Ansel Adams',
        'constraints': {
            'black_and_white': True,
            'requires_no_face': True,
            'requires_no_human': True,
            'requires_landscape': True,
            'min_contrast': 60,
            'min_sharpness': 150
        },
        'weights': {
            'contrast': 0.35,
            'sharpness': 0.25
        }
    }
    
    analyzer = StyleCompatibilityAnalyzer()
    result = analyzer.analyze_compatibility(metrics, profile)
    
    print("Résultat d'analyse:")
    print(f"Compatible: {result['compatible']}")
    print(f"Score: {result['score']:.1%}")
    print(f"Score de base: {result['base_score']:.1%}")
    print(f"Score contraintes: {result['constraint_score']:.1%}")
    
    if result['warnings']:
        print("\nAvertissements:")
        for w in result['warnings']:
            print(f"  - {w}")
    
    if result['recommendations']:
        print("\nRecommandations:")
        for r in result['recommendations']:
            print(f"  - {r}")


if __name__ == "__main__":
    example_usage()