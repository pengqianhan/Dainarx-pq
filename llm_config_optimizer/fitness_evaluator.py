"""
Fitness Evaluator: Evaluate Dainarx configurations using multiple metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class FitnessEvaluator:
    """Evaluate fitness of Dainarx configurations"""

    def __init__(self, ground_truth=None):
        """
        Initialize fitness evaluator

        Args:
            ground_truth: Optional ground truth data for evaluation
                - gt_modes: Ground truth mode labels
                - gt_changepoints: Ground truth change points
        """
        self.ground_truth = ground_truth or {}

    def evaluate(self, slice_data: List, evaluation_obj,
                 has_ground_truth: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate configuration fitness using multiple metrics

        Args:
            slice_data: List of Slice objects from Dainarx
            evaluation_obj: Evaluation object with metrics
            has_ground_truth: Whether ground truth is available

        Returns:
            (fitness_score, metrics_dict)
        """
        metrics = {}

        # 1. Mode clustering quality
        if slice_data and len(slice_data) > 0:
            metrics['num_modes'] = self._count_unique_modes(slice_data)
            metrics['valid_segments'] = sum(1 for s in slice_data if s.valid)
            metrics['invalid_segments'] = sum(1 for s in slice_data if not s.valid)
            metrics['avg_fitting_error'] = self._compute_avg_fitting_error(slice_data)
            metrics['mode_balance'] = self._compute_mode_balance(slice_data)
        else:
            # Penalize empty results
            return -1e6, {"error": "No valid slices"}

        # 2. Ground truth comparison (if available)
        if has_ground_truth and evaluation_obj:
            try:
                # Get evaluation metrics
                eval_metrics = evaluation_obj.get_metrics()

                if 'mode_accuracy' in eval_metrics:
                    metrics['mode_accuracy'] = eval_metrics['mode_accuracy']

                if 'chp_f1' in eval_metrics:
                    metrics['chp_f1'] = eval_metrics['chp_f1']
                    metrics['chp_precision'] = eval_metrics.get('chp_precision', 0)
                    metrics['chp_recall'] = eval_metrics.get('chp_recall', 0)

            except Exception as e:
                print(f"Warning: Could not extract evaluation metrics: {e}")

        # 3. Compute composite fitness
        fitness = self._compute_fitness(metrics, has_ground_truth)

        return fitness, metrics

    def _count_unique_modes(self, slice_data: List) -> int:
        """Count number of unique modes"""
        modes = set()
        for s in slice_data:
            if s.valid:
                modes.add(s.mode)
        return len(modes)

    def _compute_avg_fitting_error(self, slice_data: List) -> float:
        """Compute average fitting error across all valid segments"""
        errors = [s.err for s in slice_data if s.valid and hasattr(s, 'err')]
        if len(errors) == 0:
            return 1e6
        return float(np.mean(errors))

    def _compute_mode_balance(self, slice_data: List) -> float:
        """
        Compute mode balance metric (1.0 = perfectly balanced, 0.0 = very imbalanced)
        """
        mode_counts = {}
        for s in slice_data:
            if s.valid:
                mode_counts[s.mode] = mode_counts.get(s.mode, 0) + 1

        if len(mode_counts) == 0:
            return 0.0

        counts = list(mode_counts.values())
        # Use coefficient of variation (inverse)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count == 0:
            return 0.0

        cv = std_count / mean_count
        # Convert to balance score (0 = imbalanced, 1 = balanced)
        balance = 1.0 / (1.0 + cv)

        return float(balance)

    def _compute_fitness(self, metrics: Dict[str, float], has_ground_truth: bool) -> float:
        """
        Compute composite fitness score

        Fitness components:
        - Mode accuracy (if GT available)
        - Change point F1 (if GT available)
        - Fitting error (lower is better)
        - Model complexity penalty
        - Mode balance
        """

        if has_ground_truth:
            # With ground truth: prioritize accuracy metrics
            mode_acc = metrics.get('mode_accuracy', 0.0)
            chp_f1 = metrics.get('chp_f1', 0.0)
            fitting_err = metrics.get('avg_fitting_error', 1.0)
            num_modes = metrics.get('num_modes', 1)
            mode_balance = metrics.get('mode_balance', 0.5)

            # Normalize fitting error (assume good error < 0.1, bad > 1.0)
            fitting_score = max(0, 1.0 - min(fitting_err / 0.1, 10.0))

            # Complexity penalty (prefer 2-5 modes)
            complexity_penalty = 0.0
            if num_modes < 2:
                complexity_penalty = 0.2  # Too simple
            elif num_modes > 10:
                complexity_penalty = 0.1 * (num_modes - 10)  # Too complex

            # Weighted combination
            fitness = (
                0.35 * mode_acc +           # Mode accuracy
                0.30 * chp_f1 +              # Change point detection
                0.20 * fitting_score +       # Fitting quality
                0.10 * mode_balance -        # Mode balance
                0.05 * complexity_penalty    # Complexity
            )

        else:
            # Without ground truth: use intrinsic quality metrics
            fitting_err = metrics.get('avg_fitting_error', 1.0)
            num_modes = metrics.get('num_modes', 1)
            valid_segments = metrics.get('valid_segments', 0)
            invalid_segments = metrics.get('invalid_segments', 0)
            mode_balance = metrics.get('mode_balance', 0.5)

            # Normalize fitting error
            fitting_score = max(0, 1.0 - min(fitting_err / 0.1, 10.0))

            # Validity ratio
            total_segments = valid_segments + invalid_segments
            validity_ratio = valid_segments / total_segments if total_segments > 0 else 0.0

            # Complexity penalty
            complexity_penalty = 0.0
            if num_modes < 2:
                complexity_penalty = 0.3
            elif num_modes > 10:
                complexity_penalty = 0.1 * (num_modes - 10)

            # Weighted combination
            fitness = (
                0.40 * fitting_score +       # Fitting quality
                0.25 * validity_ratio +      # Valid segments ratio
                0.20 * mode_balance +        # Mode balance
                0.15 * min(num_modes / 5.0, 1.0) -  # Reasonable number of modes
                complexity_penalty
            )

        return float(fitness)

    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n=== Fitness Metrics ===")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
