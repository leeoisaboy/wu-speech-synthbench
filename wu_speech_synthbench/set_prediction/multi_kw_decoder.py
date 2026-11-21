from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .score_vector_baseline import LogisticCalibrator


@dataclass(frozen=True)
class SetDecision:
    sample_id: str
    selected: list[str]
    scores: list[float]
    z_scores: list[float]
    probabilities: list[float] | None = None
    rejected: bool = False


def _normalized_scores(scores: Sequence[float], method: str = "zscore") -> np.ndarray:
    """Normalize within-utterance scores via classic Z-score or robust median/MAD."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional array")
    if method == "zscore":
        std = float(values.std())
        return (values - float(values.mean())) / max(std, 1e-12)
    if method == "robust":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1e-12:
            scale = max(float(values.std()), 1e-12)
        return (values - median) / scale
    raise ValueError(f"unsupported normalization method: {method!r}")


def _average_descending_ranks(values: np.ndarray) -> np.ndarray:
    """Compute order-invariant 0-based average descending ranks for tie-safe feature extraction."""
    n = len(values)
    order = np.argsort(-values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def extract_candidate_features(
    raw_values: Sequence[float],
    z_values: Sequence[float],
) -> np.ndarray:
    """Build tie-invariant per-candidate features ``[raw_score, norm_score, gap_to_top, rank_ratio]``."""
    raw = np.asarray(raw_values, dtype=np.float64)
    norm = np.asarray(z_values, dtype=np.float64)
    top_raw = float(raw.max())
    ranks = _average_descending_ranks(raw)
    rank_ratio = ranks / max(len(raw) - 1, 1)
    gap_to_top = top_raw - raw
    return np.column_stack((raw, norm, gap_to_top, rank_ratio))


def fit_set_calibrator(
    records: Sequence[dict[str, Any]],
    normalization: str = "robust",
    l2: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    correct_prior: bool = True,
    feature_fn: Callable[[Sequence[float], Sequence[float]], np.ndarray] | None = None,
) -> LogisticCalibrator:
    """Fit a ``LogisticCalibrator`` on labeled score-vector records.

    In v3, ``class_weight`` defaults to ``None`` to preserve true posterior calibration
    (and when ``class_weight`` is set, ``correct_prior=True`` corrects the log-odds offset).
    """
    extractor = feature_fn or extract_candidate_features
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for record in records:
        expected = set(record.get("expected_phrases") or record.get("expected_targets") or [])
        phrases = [str(item.get("phrase") if isinstance(item, dict) else item[0]) for item in record["scores"]]
        raw_values = [float(item.get("score") if isinstance(item, dict) else item[1]) for item in record["scores"]]
        z_values = _normalized_scores(raw_values, method=normalization)
        feature_blocks.append(extractor(raw_values, z_values))
        label_blocks.append(np.asarray([1.0 if p in expected else 0.0 for p in phrases], dtype=np.float64))
    if not feature_blocks:
        raise ValueError("cannot fit calibrator on empty records")
    features = np.vstack(feature_blocks)
    labels = np.concatenate(label_blocks)
    return LogisticCalibrator(l2=l2, class_weight=class_weight, correct_prior=correct_prior).fit(features, labels)


def decode_set(
    record: dict[str, Any],
    cap: int = 2,
    margin: float = 1.5,
    minimum_z: float | None = None,
    normalization: str = "zscore",
    calibrator: LogisticCalibrator | None = None,
    min_probability: float | None = None,
    prob_margin: float | None = None,
    feature_fn: Callable[[Sequence[float], Sequence[float]], np.ndarray] | None = None,
) -> SetDecision:
    """Convert a phrase score vector into a bounded within-utterance set with rejection support.

    Improvements in v3:
    - When ``calibrator`` is provided, ``top_z`` is computed as ``max(z_values)`` and
      candidates failing the z-space ``margin`` check are skipped (``continue``) rather than
      prematurely terminating the probability-sorted loop (``break``).
    - Supports optional ``prob_margin`` directly in calibrated probability space.
    """
    if cap < 1:
        raise ValueError("cap must be at least one")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if min_probability is not None and not (0.0 <= min_probability <= 1.0):
        raise ValueError("min_probability must be in [0.0, 1.0]")
    if prob_margin is not None and prob_margin < 0:
        raise ValueError("prob_margin must be non-negative")

    phrases = [str(item.get("phrase") if isinstance(item, dict) else item[0]) for item in record["scores"]]
    raw_values = [float(item.get("score") if isinstance(item, dict) else item[1]) for item in record["scores"]]
    if len(set(phrases)) != len(phrases):
        raise ValueError("phrase names in one score vector must be unique")

    z_values = _normalized_scores(raw_values, method=normalization)
    probs: np.ndarray | None = None
    if calibrator is not None:
        extractor = feature_fn or extract_candidate_features
        feats = extractor(raw_values, z_values)
        probs = calibrator.predict_proba(feats)
        order = np.argsort(-probs, kind="mergesort")
        top_z = float(np.max(z_values))
        top_prob = float(probs[order[0]])
    else:
        order = np.argsort(-z_values, kind="mergesort")
        top_z = float(z_values[order[0]])
        top_prob = 1.0

    selected_indices: list[int] = []
    for index in order:
        z_val = float(z_values[index])
        if len(selected_indices) >= cap:
            break
        if probs is not None:
            prob_val = float(probs[index])
            if min_probability is not None and prob_val < min_probability:
                break
            if prob_margin is not None and (top_prob - prob_val) > prob_margin:
                break
            # Order is by probability, so z-space checks must continue rather than break
            if top_z - z_val > margin:
                continue
            if minimum_z is not None and z_val < minimum_z:
                continue
        else:
            if top_z - z_val > margin:
                break
            if minimum_z is not None and z_val < minimum_z:
                break
        selected_indices.append(int(index))

    return SetDecision(
        sample_id=str(record.get("sample_id") or record.get("audio_id") or ""),
        selected=[phrases[idx] for idx in selected_indices],
        scores=[raw_values[idx] for idx in selected_indices],
        z_scores=[float(z_values[idx]) for idx in selected_indices],
        probabilities=[float(probs[idx]) for idx in selected_indices] if probs is not None else None,
        rejected=len(selected_indices) == 0,
    )


def _metrics_for(
    records: Sequence[dict[str, Any]],
    cap: int,
    margin: float,
    minimum_z: float | None,
    normalization: str = "zscore",
    calibrator: LogisticCalibrator | None = None,
    min_probability: float | None = None,
    prob_margin: float | None = None,
) -> dict[str, float | None]:
    decisions = [
        decode_set(
            row,
            cap=cap,
            margin=margin,
            minimum_z=minimum_z,
            normalization=normalization,
            calibrator=calibrator,
            min_probability=min_probability,
            prob_margin=prob_margin,
        )
        for row in records
    ]
    truths = [set(row.get("expected_phrases") or row.get("expected_targets") or []) for row in records]
    predictions = [set(decision.selected) for decision in decisions]

    occurrences = sum(len(truth) for truth in truths)
    hits = sum(len(truth & prediction) for truth, prediction in zip(truths, predictions))
    selected_total = sum(len(prediction) for prediction in predictions)
    exact_matches = sum(truth == prediction for truth, prediction in zip(truths, predictions))

    positive_total = sum(1 for truth in truths if truth)
    recall_complete_pos = sum(
        truth.issubset(prediction) for truth, prediction in zip(truths, predictions) if truth
    )

    negative_total = sum(1 for truth in truths if not truth)
    false_alarms = sum(1 for truth, prediction in zip(truths, predictions) if not truth and len(prediction) > 0)

    recall = hits / max(occurrences, 1) if occurrences > 0 else 0.0
    precision = hits / max(selected_total, 1) if selected_total > 0 else (1.0 if occurrences == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = false_alarms / negative_total if negative_total > 0 else 0.0
    n_records = max(len(records), 1)

    return {
        "cap": float(cap),
        "margin": float(margin),
        "minimum_z": None if minimum_z is None else float(minimum_z),
        "min_probability": None if min_probability is None else float(min_probability),
        "occurrence_recall": recall,
        "precision": precision,
        "f1": f1,
        "exact_set_match": exact_matches / n_records,
        "sentence_recall_complete": (recall_complete_pos / positive_total) if positive_total > 0 else 0.0,
        "false_alarm_rate": far,
        "negative_rejection_rate": (1.0 - far) if negative_total > 0 else None,
    }


def sweep_set_decisions(
    records: Sequence[dict[str, Any]],
    caps: Sequence[int] = (1, 2, 3),
    margins: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    minimum_zs: Sequence[float | None] = (None,),
    normalization: str = "zscore",
    calibrator: LogisticCalibrator | None = None,
    min_probabilities: Sequence[float | None] = (None,),
    prob_margin: float | None = None,
) -> list[dict[str, float | None]]:
    """Report the deterministic operating surface across cap, margin, z-floor, and probability thresholds."""
    return [
        _metrics_for(
            records,
            cap=cap,
            margin=margin,
            minimum_z=min_z,
            normalization=normalization,
            calibrator=calibrator,
            min_probability=min_prob,
            prob_margin=prob_margin,
        )
        for cap in caps
        for margin in margins
        for min_z in minimum_zs
        for min_prob in min_probabilities
    ]
