from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std < 1e-12 or right_std < 1e-12:
        return 0.0
    return float(np.mean((left - left.mean()) * (right - right.mean())) / (left_std * right_std))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute average ranks (1-based) with proper tie handling in pure NumPy."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    return _pearson(_average_ranks(left), _average_ranks(right))


def frequency_bias(word_recalls: dict[str, float], word_counts: dict[str, int]) -> dict[str, float]:
    """Correlate word recall with raw/log occurrence frequency via Pearson and Spearman."""
    words = sorted(set(word_recalls) & set(word_counts))
    if not words:
        return {
            "pearson_rho_raw_frequency": 0.0,
            "pearson_rho_log_frequency": 0.0,
            "spearman_rho_raw_frequency": 0.0,
            "spearman_rho_log_frequency": 0.0,
            "word_count": 0,
        }
    recalls = np.asarray([word_recalls[word] for word in words], dtype=np.float64)
    counts = np.asarray([word_counts[word] for word in words], dtype=np.float64)
    log_counts = np.log1p(counts)
    return {
        "pearson_rho_raw_frequency": _pearson(counts, recalls),
        "pearson_rho_log_frequency": _pearson(log_counts, recalls),
        "spearman_rho_raw_frequency": _spearman(counts, recalls),
        "spearman_rho_log_frequency": _spearman(log_counts, recalls),
        "word_count": len(words),
    }


def _average_precision_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute tie-invariant PR-AUC (Average Precision) grouped by distinct score thresholds.

    Groups identical scores together so the result is completely invariant to input ordering
    (matching standard ``sklearn.metrics.average_precision_score`` threshold semantics).
    """
    total_pos = int(np.sum(labels == 1))
    if total_pos == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ap = 0.0
    cum_tp = 0
    cum_count = 0
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        group_tp = int(np.sum(sorted_labels[i:j] == 1))
        cum_tp += group_tp
        cum_count += j - i
        if group_tp > 0:
            precision_at_threshold = cum_tp / float(cum_count)
            delta_recall = group_tp / float(total_pos)
            ap += precision_at_threshold * delta_recall
        i = j
    return float(ap)


def _extract_frequency_map(
    lexicon: Sequence[Any] | Mapping[str, str] | None,
) -> dict[str, str]:
    if lexicon is None:
        return {}
    if isinstance(lexicon, Mapping):
        return {str(k): str(v) for k, v in lexicon.items()}
    mapping: dict[str, str] = {}
    for entry in lexicon:
        if isinstance(entry, Mapping):
            phrase = str(entry.get("phrase") or "")
            freq = str(entry.get("frequency_class") or "tail")
        else:
            phrase = str(getattr(entry, "phrase", ""))
            freq = str(getattr(entry, "frequency_class", "tail"))
        if phrase:
            mapping[phrase] = freq
    return mapping


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def evaluate_ranking(
    records: Sequence[dict[str, Any]],
    top_k: int = 1,
    lexicon: Sequence[Any] | Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report per-occurrence, per-word, frequency-class, ranking, and negative-rejection metrics.

    Improvements in v3:
    - ``sentence_recall_complete`` is computed strictly over positive utterances
      (denominator = ``positive_records``) so negative slices cannot inflate a 0.50 floor.
    - ``pr_auc`` is computed via tie-invariant threshold grouping.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least one")

    freq_map = _extract_frequency_map(lexicon)
    occurrence_hits = 0
    occurrence_total = 0
    word_hits: dict[str, int] = defaultdict(int)
    word_total: dict[str, int] = defaultdict(int)
    positive_records = 0
    recall_complete_positives = 0
    exact_match_sentences = 0

    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    ndcgs: list[float] = []

    negative_total = 0
    negative_false_alarms = 0
    subset_neg_total: dict[str, int] = defaultdict(int)
    subset_neg_fa: dict[str, int] = defaultdict(int)

    pr_scores: list[float] = []
    pr_labels: list[int] = []

    for row in records:
        raw_expected = row.get("expected_phrases")
        if raw_expected is None:
            raw_expected = row.get("expected_targets") or []
        expected = list(raw_expected)
        expected_set = set(expected)

        full_predicted = _dedupe_preserve_order([str(p) for p in (row.get("predicted_phrases") or [])])
        predicted_k = full_predicted[:top_k]
        predicted_set = set(predicted_k)

        if "scores" in row and row["scores"]:
            for item in row["scores"]:
                phrase = str(item.get("phrase") if isinstance(item, dict) else item[0])
                score = float(item.get("score") if isinstance(item, dict) else item[1])
                pr_scores.append(score)
                pr_labels.append(1 if phrase in expected_set else 0)

        if not expected:
            negative_total += 1
            subset = str(row.get("subset") or "negative")
            subset_neg_total[subset] += 1
            if predicted_k:
                negative_false_alarms += 1
                subset_neg_fa[subset] += 1
            exact_match_sentences += int(len(predicted_set) == 0)
            continue

        positive_records += 1
        for phrase in expected:
            occurrence_total += 1
            word_total[phrase] += 1
            hit = phrase in predicted_set
            occurrence_hits += int(hit)
            word_hits[phrase] += int(hit)

        recall_complete_positives += int(expected_set.issubset(predicted_set))
        exact_match_sentences += int(expected_set == predicted_set)

        # MRR (over full ranked list)
        rr = 0.0
        for rank_idx, phrase in enumerate(full_predicted, start=1):
            if phrase in expected_set:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

        # MAP (over full ranked list)
        hits_so_far = 0
        ap_sum = 0.0
        for rank_idx, phrase in enumerate(full_predicted, start=1):
            if phrase in expected_set:
                hits_so_far += 1
                ap_sum += hits_so_far / rank_idx
        average_precisions.append(ap_sum / len(expected_set))

        # NDCG@K (binary relevance)
        dcg = sum(
            1.0 / np.log2(rank_idx + 1.0)
            for rank_idx, phrase in enumerate(predicted_k, start=1)
            if phrase in expected_set
        )
        ideal_hits = min(len(expected_set), top_k)
        idcg = sum(1.0 / np.log2(rank_idx + 1.0) for rank_idx in range(1, ideal_hits + 1))
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

    word_recalls = {word: word_hits[word] / word_total[word] for word in word_total}
    counts = {word: int(word_total[word]) for word in word_total}

    stratified: dict[str, dict[str, Any]] = {}
    if freq_map and word_total:
        by_class: dict[str, list[str]] = defaultdict(list)
        for word in word_total:
            by_class[freq_map.get(word, "unmapped")].append(word)
        for cls_name, words in sorted(by_class.items()):
            cls_hits = sum(word_hits[w] for w in words)
            cls_total = sum(word_total[w] for w in words)
            stratified[cls_name] = {
                "word_count": len(words),
                "occurrences": int(cls_total),
                "micro_recall": cls_hits / max(cls_total, 1),
                "macro_recall": float(np.mean([word_recalls[w] for w in words])),
            }

    n_records = len(records)
    far = negative_false_alarms / negative_total if negative_total > 0 else 0.0
    subset_far = {
        subset: subset_neg_fa[subset] / total
        for subset, total in sorted(subset_neg_total.items())
        if total > 0
    }
    pr_auc = (
        _average_precision_binary(np.asarray(pr_scores, dtype=np.float64), np.asarray(pr_labels, dtype=np.int64))
        if pr_scores
        else None
    )

    return {
        "records": n_records,
        "positive_records": positive_records,
        "negative_records": negative_total,
        "top_k": top_k,
        "micro_top_k": occurrence_hits / occurrence_total if occurrence_total > 0 else 0.0,
        "macro_top_k": float(np.mean(list(word_recalls.values()))) if word_recalls else 0.0,
        "sentence_recall_complete": (recall_complete_positives / positive_records) if positive_records > 0 else 0.0,
        "exact_set_match": exact_match_sentences / max(n_records, 1) if n_records else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "map": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "false_alarm_rate": far,
        "subset_false_alarm_rates": subset_far,
        "pr_auc": pr_auc,
        "frequency_class_recalls": stratified,
        "word_recalls": word_recalls,
        "word_counts": counts,
        "frequency_bias": frequency_bias(word_recalls, counts),
    }
