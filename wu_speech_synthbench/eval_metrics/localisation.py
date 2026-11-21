from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np


def iou(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def _validate_span_bounds(start: float, end: float, label: str, sample_id: str) -> None:
    if end < start:
        raise ValueError(f"{label} end ({end}) < start ({start}) for sample {sample_id!r}")


def _max_weight_bipartite_matching(
    weights: np.ndarray,
    include_zero: bool = False,
) -> list[tuple[int, int, float]]:
    """Exact maximum-weight bipartite matching via Hungarian algorithm (pure NumPy)."""
    n_rows, n_cols = weights.shape
    if n_rows == 0 or n_cols == 0:
        return []
    transposed = False
    mat = np.asarray(weights, dtype=np.float64)
    if n_rows > n_cols:
        mat = mat.T
        n_rows, n_cols = mat.shape
        transposed = True

    max_val = float(np.max(mat))
    cost = max_val - mat

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int, float]] = []
    for j in range(1, n_cols + 1):
        i = int(p[j])
        if i > 0:
            w = float(mat[i - 1, j - 1])
            if include_zero or w > 0.0:
                if transposed:
                    pairs.append((j - 1, i - 1, w))
                else:
                    pairs.append((i - 1, j - 1, w))
    return pairs


def evaluate_localisation(
    records: Sequence[dict[str, Any]],
    strict_contract: bool = True,
) -> dict[str, Any]:
    """Separate phrase-selection quality from temporal-overlap quality with contract enforcement."""
    end_to_end: list[float] = []
    known_word: list[float] = []
    correct_words = 0
    positionable = 0
    sources: Counter[str] = Counter()
    contract_violations: list[dict[str, Any]] = []

    for idx, row in enumerate(records):
        sample_id = str(row.get("sample_id") or f"row-{idx}")
        expected_start = float(row["expected_start"])
        expected_end = float(row["expected_end"])
        _validate_span_bounds(expected_start, expected_end, "expected", sample_id)

        selected = str(row["predicted_phrase"])
        expected_phrase = str(row["expected_phrase"])
        correct_words += int(selected == expected_phrase)
        source = str(row.get("char_source") or "unknown")
        sources[source] += 1

        pred_start = row.get("predicted_start")
        pred_end = row.get("predicted_end")

        if source == "no-position" and (pred_start is not None or pred_end is not None):
            violation = {
                "sample_id": sample_id,
                "reason": "no-position record carries non-null predicted coordinates",
                "predicted_start": pred_start,
                "predicted_end": pred_end,
            }
            contract_violations.append(violation)
            if strict_contract:
                raise ValueError(
                    f"Data contract violation in {sample_id!r}: char_source='no-position' "
                    f"must have predicted_start=None and predicted_end=None"
                )
            pred_start = None
            pred_end = None

        has_both_bounds = pred_start is not None and pred_end is not None
        if not has_both_bounds:
            if (pred_start is None) != (pred_end is None):
                violation = {
                    "sample_id": sample_id,
                    "reason": "half-missing predicted span bounds",
                    "predicted_start": pred_start,
                    "predicted_end": pred_end,
                }
                contract_violations.append(violation)
                if strict_contract:
                    raise ValueError(
                        f"Data contract violation in {sample_id!r}: predicted_start and predicted_end "
                        f"must both be null or both be numeric"
                    )
            end_to_end.append(0.0)
        else:
            p_start = float(pred_start)
            p_end = float(pred_end)
            _validate_span_bounds(p_start, p_end, "predicted", sample_id)
            positionable += 1
            overlap = iou(expected_start, expected_end, p_start, p_end)
            end_to_end.append(overlap)
            if selected == expected_phrase:
                known_word.append(overlap)

    count = len(records)
    return {
        "records": count,
        "phrase_selection_accuracy": correct_words / max(count, 1) if count else 0.0,
        "end_to_end_iou": sum(end_to_end) / max(count, 1) if count else 0.0,
        "known_word_iou": sum(known_word) / max(len(known_word), 1) if known_word else 0.0,
        "known_word_positionable_count": len(known_word),
        "positionable_count": positionable,
        "char_source_counts": dict(sources),
        "contract_violations": contract_violations,
    }


def evaluate_multi_span_localisation(
    records: Sequence[dict[str, Any]],
    iou_threshold: float = 0.5,
    strict_contract: bool = True,
) -> dict[str, Any]:
    """Evaluate multi-phrase long-utterance localisation via exact maximum-weight bipartite matching.

    Improvements in v3:
    - Includes same-phrase zero-overlap pairs (IoU == 0.0) in ``multi_span_known_word_iou``
      so off-target boundaries on correctly selected words are penalized consistently with
      single-span ``evaluate_localisation``.
    - Tracks both ``total_predicted_span_count`` (including ``no-position`` predictions) and
      ``positionable_predicted_span_count``, using ``total_predicted_span_count`` as the
      end-to-end ``precision_at_iou`` denominator so unpositionable false alarms cannot
      inflate precision.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0.0, 1.0]")

    total_expected = 0
    total_predicted_all = 0
    total_predicted_positionable = 0
    matched_tp_at_iou = 0
    end_to_end_ious: list[float] = []
    known_word_ious: list[float] = []
    sources: Counter[str] = Counter()
    contract_violations: list[dict[str, Any]] = []

    for idx, row in enumerate(records):
        sample_id = str(row.get("sample_id") or f"row-{idx}")
        exp_spans = list(row.get("expected_spans") or [])
        pred_spans_raw = list(row.get("predicted_spans") or [])
        total_expected += len(exp_spans)
        total_predicted_all += len(pred_spans_raw)

        valid_preds: list[dict[str, Any]] = []
        for pred in pred_spans_raw:
            source = str(pred.get("char_source") or "unknown")
            sources[source] += 1
            p_start = pred.get("start")
            p_end = pred.get("end")
            if source == "no-position" and (p_start is not None or p_end is not None):
                contract_violations.append(
                    {"sample_id": sample_id, "reason": "no-position span carries non-null coordinates"}
                )
                if strict_contract:
                    raise ValueError(f"Data contract violation in {sample_id!r}: no-position span has coordinates")
                continue
            if p_start is None or p_end is None:
                if (p_start is None) != (p_end is None):
                    contract_violations.append(
                        {"sample_id": sample_id, "reason": "half-missing predicted span bounds"}
                    )
                    if strict_contract:
                        raise ValueError(f"Data contract violation in {sample_id!r}: half-missing predicted bounds")
                continue
            _validate_span_bounds(float(p_start), float(p_end), "predicted", sample_id)
            valid_preds.append(pred)

        total_predicted_positionable += len(valid_preds)
        if not exp_spans:
            continue

        for exp in exp_spans:
            _validate_span_bounds(float(exp["start"]), float(exp["end"]), "expected", sample_id)

        if not valid_preds:
            end_to_end_ious.extend([0.0] * len(exp_spans))
            continue

        # 1. End-to-end optimal matching (phrase-agnostic overlap per expected span)
        e2e_weights = np.zeros((len(exp_spans), len(valid_preds)), dtype=np.float64)
        for i, exp in enumerate(exp_spans):
            e_start, e_end = float(exp["start"]), float(exp["end"])
            for j, pred in enumerate(valid_preds):
                e2e_weights[i, j] = iou(e_start, e_end, float(pred["start"]), float(pred["end"]))

        e2e_matches = _max_weight_bipartite_matching(e2e_weights, include_zero=False)
        matched_e2e_by_exp = {i: w for i, _, w in e2e_matches}
        for i in range(len(exp_spans)):
            end_to_end_ious.append(matched_e2e_by_exp.get(i, 0.0))

        # 2. Known-word optimal matching grouped by phrase (includes zero-overlap same-phrase pairs!)
        exp_by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pred_by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for exp in exp_spans:
            exp_by_phrase[str(exp["phrase"])].append(exp)
        for pred in valid_preds:
            pred_by_phrase[str(pred["phrase"])].append(pred)

        for phrase, p_exps in exp_by_phrase.items():
            p_preds = pred_by_phrase.get(phrase)
            if not p_preds:
                continue
            mat = np.zeros((len(p_exps), len(p_preds)), dtype=np.float64)
            for i, exp in enumerate(p_exps):
                e_start, e_end = float(exp["start"]), float(exp["end"])
                for j, pred in enumerate(p_preds):
                    mat[i, j] = iou(e_start, e_end, float(pred["start"]), float(pred["end"]))
            # include_zero=True ensures min(len(p_exps), len(p_preds)) same-phrase pairs are all scored
            kw_matches = _max_weight_bipartite_matching(mat, include_zero=True)
            for _, _, w in kw_matches:
                known_word_ious.append(w)
                if w >= iou_threshold:
                    matched_tp_at_iou += 1

    precision_at_iou = matched_tp_at_iou / total_predicted_all if total_predicted_all > 0 else 0.0
    positionable_precision_at_iou = (
        matched_tp_at_iou / total_predicted_positionable if total_predicted_positionable > 0 else 0.0
    )
    recall_at_iou = matched_tp_at_iou / total_expected if total_expected > 0 else 0.0
    f1_at_iou = (
        2.0 * precision_at_iou * recall_at_iou / (precision_at_iou + recall_at_iou)
        if (precision_at_iou + recall_at_iou) > 0
        else 0.0
    )

    return {
        "records": len(records),
        "iou_threshold": iou_threshold,
        "expected_span_count": total_expected,
        "total_predicted_span_count": total_predicted_all,
        "positionable_predicted_span_count": total_predicted_positionable,
        "matched_at_iou_count": matched_tp_at_iou,
        "known_word_matched_pair_count": len(known_word_ious),
        "multi_span_end_to_end_iou": float(np.mean(end_to_end_ious)) if end_to_end_ious else 0.0,
        "multi_span_known_word_iou": float(np.mean(known_word_ious)) if known_word_ious else 0.0,
        "precision_at_iou": precision_at_iou,
        "positionable_precision_at_iou": positionable_precision_at_iou,
        "recall_at_iou": recall_at_iou,
        "f1_at_iou": f1_at_iou,
        "char_source_counts": dict(sources),
        "contract_violations": contract_violations,
    }
