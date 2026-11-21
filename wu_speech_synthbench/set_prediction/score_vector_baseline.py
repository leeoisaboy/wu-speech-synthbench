from __future__ import annotations

from typing import Mapping

import numpy as np


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    if probs.shape != target.shape:
        raise ValueError("probabilities and labels must have the same shape")
    return float(np.mean((probs - target) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform confidence bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    if probs.shape != target.shape:
        raise ValueError("probabilities and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(probs))
    for bin_idx in range(n_bins):
        low, high = edges[bin_idx], edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(probs[mask]))
        avg_acc = float(np.mean(target[mask]))
        ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


class LogisticCalibrator:
    """Deterministic ridge-regularized logistic calibrator with Armijo line search & prior correction.

    Improvements in v3:
    - When trained with non-uniform ``class_weight`` (e.g., ``"balanced"``), records
      ``log_prior_offset = ln(w1 / w0)`` and subtracts it at inference time
      (``correct_prior=True`` by default) so ``predict_proba`` returns calibrated
      true posterior probabilities ``P(y=1 | x)`` without artificial 50/50 prior drift.
    """

    def __init__(
        self,
        l2: float = 1.0,
        iterations: int = 50,
        class_weight: str | Mapping[int, float] | None = None,
        correct_prior: bool = True,
        armijo_c: float = 1e-4,
        backtrack_beta: float = 0.5,
        max_backtracks: int = 20,
    ):
        if l2 <= 0 or iterations < 1:
            raise ValueError("l2 must be positive and iterations at least one")
        self.l2 = float(l2)
        self.iterations = int(iterations)
        self.class_weight = class_weight
        self.correct_prior = bool(correct_prior)
        self.armijo_c = float(armijo_c)
        self.backtrack_beta = float(backtrack_beta)
        self.max_backtracks = int(max_backtracks)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.log_prior_offset: float = 0.0

    def _resolve_sample_weights(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        n = len(target)
        if self.class_weight is None:
            return np.ones(n, dtype=np.float64), 0.0
        if self.class_weight == "balanced":
            n_pos = float(np.sum(target == 1.0))
            n_neg = float(np.sum(target == 0.0))
            w_pos = n / (2.0 * max(n_pos, 1.0))
            w_neg = n / (2.0 * max(n_neg, 1.0))
            return np.where(target == 1.0, w_pos, w_neg), float(np.log(w_pos / w_neg))
        if isinstance(self.class_weight, Mapping):
            w_neg = float(self.class_weight.get(0, 1.0))
            w_pos = float(self.class_weight.get(1, 1.0))
            if w_neg <= 0 or w_pos <= 0:
                raise ValueError("class_weight values must be positive")
            return np.where(target == 1.0, w_pos, w_neg), float(np.log(w_pos / w_neg))
        raise ValueError(f"unsupported class_weight: {self.class_weight!r}")

    @staticmethod
    def _objective(
        design: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        sample_prior_w: np.ndarray,
        penalty: np.ndarray,
    ) -> float:
        linear = design @ weights
        log1p_exp = np.maximum(linear, 0.0) + np.log1p(np.exp(-np.abs(linear)))
        nll = float(np.sum(sample_prior_w * (log1p_exp - target * linear)))
        reg = 0.5 * float(weights @ (penalty @ weights))
        return nll + reg

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticCalibrator":
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
            raise ValueError("features must be [samples, dimensions] and labels [samples]")
        if len(matrix) == 0:
            raise ValueError("features must not be empty")
        if set(np.unique(target)) - {0.0, 1.0}:
            raise ValueError("labels must contain only zero and one")

        self._mean = matrix.mean(axis=0)
        self._scale = np.maximum(matrix.std(axis=0), 1e-12)
        design = np.column_stack((np.ones(len(matrix)), (matrix - self._mean) / self._scale))
        n_params = design.shape[1]
        weights = np.zeros(n_params, dtype=np.float64)
        penalty = np.diag(np.r_[0.0, np.full(n_params - 1, self.l2)])
        prior_w, log_offset = self._resolve_sample_weights(target)
        self.log_prior_offset = log_offset
        eye = np.eye(n_params, dtype=np.float64)

        for _ in range(self.iterations):
            linear = design @ weights
            probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))
            hess_weight = prior_w * np.maximum(probability * (1.0 - probability), 1e-9)
            gradient = design.T @ (prior_w * (probability - target)) + penalty @ weights
            hessian = design.T @ (design * hess_weight[:, None]) + penalty

            step: np.ndarray | None = None
            for damping in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    candidate_step = np.linalg.solve(hessian + damping * eye, gradient)
                    if np.all(np.isfinite(candidate_step)):
                        step = candidate_step
                        break
                except np.linalg.LinAlgError:
                    continue
            if step is None:
                step = np.linalg.lstsq(hessian + 1e-4 * eye, gradient, rcond=None)[0]

            directional_deriv = float(gradient @ step)
            if directional_deriv <= 0.0:
                step = gradient
                directional_deriv = float(gradient @ step)

            current_obj = self._objective(design, target, weights, prior_w, penalty)
            alpha = 1.0
            accepted = False
            for _ in range(self.max_backtracks):
                trial_weights = weights - alpha * step
                trial_obj = self._objective(design, target, trial_weights, prior_w, penalty)
                if trial_obj <= current_obj - self.armijo_c * alpha * directional_deriv:
                    weights = trial_weights
                    accepted = True
                    break
                alpha *= self.backtrack_beta
            if not accepted:
                weights = weights - alpha * step

            if float(np.max(np.abs(alpha * step))) < 1e-8:
                break

        self.weights = weights
        return self

    def predict_proba(self, features: np.ndarray, apply_prior_correction: bool | None = None) -> np.ndarray:
        if self.weights is None or self._mean is None or self._scale is None:
            raise ValueError("fit must be called before predict_proba")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        standardized = (matrix - self._mean) / self._scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        linear = design @ self.weights
        use_correction = self.correct_prior if apply_prior_correction is None else apply_prior_correction
        if use_correction and self.log_prior_offset != 0.0:
            linear = linear - self.log_prior_offset
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))

    def evaluate_calibration(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float]:
        probs = self.predict_proba(features)
        target = np.asarray(labels, dtype=np.float64)
        return {
            "ece": expected_calibration_error(probs, target, n_bins=n_bins),
            "brier_score": brier_score(probs, target),
        }
