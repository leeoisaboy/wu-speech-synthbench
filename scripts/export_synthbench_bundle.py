"""Export static JSON bundle for SynthBench Studio using real wu_speech_synthbench modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wu_speech_synthbench.auto_timestamp.energy_valley import acoustic_features, segment_word
from wu_speech_synthbench.cli import _build_synthetic_eval_slice
from wu_speech_synthbench.eval_metrics.localisation import evaluate_multi_span_localisation
from wu_speech_synthbench.eval_metrics.ranking import evaluate_ranking, frequency_bias
from wu_speech_synthbench.set_prediction.multi_kw_decoder import (
    _normalized_scores,
    decode_set,
    extract_candidate_features,
    fit_set_calibrator,
    sweep_set_decisions,
)
from wu_speech_synthbench.set_prediction.score_vector_baseline import (
    brier_score,
    expected_calibration_error,
)
from wu_speech_synthbench.synth_pipeline.demo import BACKGROUNDS, CONFUSIONS, WORDS, demo_lexicon, demo_plan
from wu_speech_synthbench.synth_pipeline.manifest import (
    BenchmarkPlan,
    ConfusionPair,
    LexiconEntry,
    build_benchmark_plan,
    validate_plan,
)


def _round_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    return [round(float(x), decimals) for x in arr]


def _make_dsp_presets() -> list[dict[str, Any]]:
    """Build two realistic synthetic speech signals with voiceless fricatives/aspirates + digital silence."""
    sr = 8000  # 8 kHz keeps JSON compact while cleanly covering 300-3400 Hz speech & 2000-4000 Hz high band
    rng = np.random.default_rng(42)

    presets: list[dict[str, Any]] = []

    # Preset 1: "烧水" — voiceless fricative 'sh' (2400-3600 Hz) + vowel 'ao' + fricative 'sh' + vowel 'ui' + silence edges
    dur1 = 1.00
    t1 = np.arange(int(sr * dur1)) / sr
    noise1 = rng.normal(size=len(t1))
    # Simple high-pass noise for voiceless fricative (2200-3700 Hz via modulation)
    fric1 = np.sin(2 * np.pi * 2900 * t1) * (0.55 + 0.45 * rng.normal(size=len(t1))) * 0.42
    fric1 += np.sin(2 * np.pi * 3450 * t1) * (0.5 + 0.5 * rng.normal(size=len(t1))) * 0.30
    vowel_a = 0.65 * np.sin(2 * np.pi * 520 * t1) + 0.38 * np.sin(2 * np.pi * 1180 * t1) + 0.16 * np.sin(2 * np.pi * 2300 * t1)
    vowel_u = 0.60 * np.sin(2 * np.pi * 430 * t1) + 0.35 * np.sin(2 * np.pi * 960 * t1) + 0.14 * np.sin(2 * np.pi * 2100 * t1)

    env_sh1 = np.exp(-0.5 * ((t1 - 0.24) / 0.032) ** 2)
    env_ao = np.exp(-0.5 * ((t1 - 0.34) / 0.052) ** 2)
    env_sh2 = np.exp(-0.5 * ((t1 - 0.54) / 0.030) ** 2)
    env_ui = np.exp(-0.5 * ((t1 - 0.65) / 0.058) ** 2)
    sig1 = fric1 * (0.85 * env_sh1 + 0.78 * env_sh2) + vowel_a * env_ao + vowel_u * env_ui
    # Hard digital silence at edges (-120 dB) to demonstrate Global vs Local Min-Max dilution!
    sig1[(t1 < 0.14) | (t1 > 0.84)] = 0.0
    sig1 = np.clip(sig1 / max(float(np.max(np.abs(sig1))), 1e-6) * 0.92, -1.0, 1.0)

    # Synthetic 2-class posteriors (50 Hz, 20ms hop, crossing right around 0.48s)
    post_hop = 0.020
    n_post1 = int(round(dur1 / post_hop))
    p_times1 = np.arange(n_post1) * post_hop
    p0 = 1.0 / (1.0 + np.exp((p_times1 - 0.48) / 0.018))
    p1 = 1.0 - p0
    post1 = np.column_stack([p0, p1])

    word1 = segment_word(sig1, sr, "烧水", [0.33, 0.64], posterior_probs=post1, posterior_hop_seconds=post_hop)
    feats1 = acoustic_features(sig1, sr)
    presets.append(
        {
            "id": "shaoshui",
            "label": "烧水（清擦音 sh + 句首尾 -120 dB 静音）",
            "characters": "烧水",
            "sample_rate": sr,
            "duration": dur1,
            "anchor_times": [0.33, 0.64],
            "signal": _round_list(sig1, 4),
            "posterior_hop": post_hop,
            "posteriors": [_round_list(post1[:, 0], 4), _round_list(post1[:, 1], 4)],
            "py_reference": {
                "word_start": round(word1.start_time, 4),
                "word_end": round(word1.end_time, 4),
                "characters": [
                    {
                        "character": c.character,
                        "start_time": round(c.start_time, 4),
                        "end_time": round(c.end_time, 4),
                    }
                    for c in word1.characters
                ],
                "n_frames": int(len(feats1["energy_db"])),
            },
        }
    )

    # Preset 2: "开灯" — aspirated stop 'k' burst + vowel 'ai' + stop 'd' + vowel 'eng'
    dur2 = 1.00
    t2 = np.arange(int(sr * dur2)) / sr
    asp2 = np.sin(2 * np.pi * 2750 * t2) * (0.5 + 0.5 * rng.normal(size=len(t2))) * 0.48
    asp2 += np.sin(2 * np.pi * 3500 * t2) * (0.5 + 0.5 * rng.normal(size=len(t2))) * 0.34
    v_ai = 0.68 * np.sin(2 * np.pi * 680 * t2) + 0.40 * np.sin(2 * np.pi * 1550 * t2)
    v_eng = 0.62 * np.sin(2 * np.pi * 490 * t2) + 0.32 * np.sin(2 * np.pi * 1240 * t2)

    env_k = np.exp(-0.5 * ((t2 - 0.21) / 0.022) ** 2)
    env_ai = np.exp(-0.5 * ((t2 - 0.33) / 0.056) ** 2)
    env_d = np.exp(-0.5 * ((t2 - 0.55) / 0.018) ** 2)
    env_eng = np.exp(-0.5 * ((t2 - 0.66) / 0.062) ** 2)
    sig2 = asp2 * (0.95 * env_k + 0.45 * env_d) + v_ai * env_ai + v_eng * env_eng
    sig2[(t2 < 0.12) | (t2 > 0.85)] = 0.0
    sig2 = np.clip(sig2 / max(float(np.max(np.abs(sig2))), 1e-6) * 0.92, -1.0, 1.0)

    n_post2 = int(round(dur2 / post_hop))
    p_times2 = np.arange(n_post2) * post_hop
    q0 = 1.0 / (1.0 + np.exp((p_times2 - 0.50) / 0.018))
    q1 = 1.0 - q0
    post2 = np.column_stack([q0, q1])

    word2 = segment_word(sig2, sr, "开灯", [0.33, 0.66], posterior_probs=post2, posterior_hop_seconds=post_hop)
    presets.append(
        {
            "id": "kaideng",
            "label": "开灯（送气清塞音 k + 鼻韵母 eng）",
            "characters": "开灯",
            "sample_rate": sr,
            "duration": dur2,
            "anchor_times": [0.33, 0.66],
            "signal": _round_list(sig2, 4),
            "posterior_hop": post_hop,
            "posteriors": [_round_list(post2[:, 0], 4), _round_list(post2[:, 1], 4)],
            "py_reference": {
                "word_start": round(word2.start_time, 4),
                "word_end": round(word2.end_time, 4),
                "characters": [
                    {
                        "character": c.character,
                        "start_time": round(c.start_time, 4),
                        "end_time": round(c.end_time, 4),
                    }
                    for c in word2.characters
                ],
            },
        }
    )

    return presets


def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for idx in range(n_bins):
        low, high = float(edges[idx]), float(edges[idx + 1])
        mask = (probs >= low) & (probs <= high if idx == n_bins - 1 else probs < high)
        count = int(np.sum(mask))
        bins.append(
            {
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "center": round(0.5 * (low + high), 2),
                "count": count,
                "avg_conf": round(float(np.mean(probs[mask])), 4) if count > 0 else None,
                "avg_acc": round(float(np.mean(labels[mask])), 4) if count > 0 else None,
            }
        )
    return bins


def _make_calibration_and_decoder_data() -> dict[str, Any]:
    """Run real LogisticCalibrator & sweep_set_decisions on independent splits."""
    cal_plan = build_benchmark_plan(
        demo_lexicon(),
        [ConfusionPair(t, c) for t, c in CONFUSIONS],
        BACKGROUNDS,
        target_count=80,
        hard_negative_count=60,
        background_count=40,
        multi_phrase_ratio=0.35,
        seed=101,
    )
    eval_plan = build_benchmark_plan(
        demo_lexicon(),
        [ConfusionPair(t, c) for t, c in CONFUSIONS],
        BACKGROUNDS,
        target_count=80,
        hard_negative_count=60,
        background_count=40,
        multi_phrase_ratio=0.35,
        seed=17,
    )
    _, _, _, cal_sets = _build_synthetic_eval_slice(cal_plan, seed=101)
    _, _, _, eval_sets = _build_synthetic_eval_slice(eval_plan, seed=17)

    # Make realistic overlap so calibration bins span [0, 1] smoothly
    rng = np.random.default_rng(77)
    for split in (cal_sets, eval_sets):
        for row in split:
            exp = set(row["expected_phrases"])
            for item in row["scores"]:
                is_pos = item["phrase"] in exp
                item["score"] = float(rng.normal(1.55 if is_pos else 0.15, 0.85))

    cal_bal = fit_set_calibrator(cal_sets, normalization="robust", class_weight="balanced", correct_prior=True)
    cal_plain = fit_set_calibrator(cal_sets, normalization="robust", class_weight=None)

    feat_blocks, label_blocks = [], []
    for row in eval_sets:
        exp = set(row["expected_phrases"])
        raw = [float(item["score"]) for item in row["scores"]]
        z = _normalized_scores(raw, method="robust")
        feat_blocks.append(extract_candidate_features(raw, z))
        label_blocks.append(np.asarray([1.0 if item["phrase"] in exp else 0.0 for item in row["scores"]]))
    X_eval = np.vstack(feat_blocks)
    y_eval = np.concatenate(label_blocks)

    probs_uncorrected = cal_bal.predict_proba(X_eval, apply_prior_correction=False)
    probs_corrected = cal_bal.predict_proba(X_eval, apply_prior_correction=True)
    probs_plain = cal_plain.predict_proba(X_eval)

    # Operating surface sweep across cap, margin, min_probability
    surface = sweep_set_decisions(
        eval_sets,
        caps=(1, 2, 3, 4),
        margins=(0.5, 1.0, 1.5, 2.0, 2.5),
        minimum_zs=(None,),
        normalization="robust",
        calibrator=cal_bal,
        min_probabilities=(0.15, 0.30, 0.45, 0.60, 0.75, 0.85),
    )

    # Curated single-utterance interactive examples
    utterances = [
        {
            "id": "multi_phrase",
            "title": "多词长句：「先去开灯，回头再拉窗帘。」",
            "expected": ["开灯", "拉窗帘"],
            "note": "拖动下方「干扰词尖峰」推子拉高「放音乐」分数，观察普通 Z-score 被带歪而 Robust MAD 稳如泰山。",
            "candidates": [
                {"phrase": "开灯", "raw": 2.38, "prob": 0.91},
                {"phrase": "拉窗帘", "raw": 2.12, "prob": 0.84},
                {"phrase": "关窗", "raw": 0.45, "prob": 0.14},
                {"phrase": "烧水", "raw": 0.31, "prob": 0.10},
                {"phrase": "调温度", "raw": 0.18, "prob": 0.08},
                {"phrase": "开风扇", "raw": 0.09, "prob": 0.06},
                {"phrase": "放音乐", "raw": 0.22, "prob": 0.09, "spike_target": True},
                {"phrase": "扫地", "raw": -0.05, "prob": 0.04},
            ],
        },
        {
            "id": "continue_rescue",
            "title": "排序分歧救援：prob 排序 ≠ z-score 排序（continue vs break）",
            "expected": ["开灯", "烧水"],
            "note": "「调温度」因校准概率排第 2 但 z-score 落后较多：旧版提前 break 会连带误杀排第 3 的真目标「烧水」；v3 用 continue 精准救回！",
            "candidates": [
                {"phrase": "开灯", "raw": 2.65, "prob": 0.93},
                {"phrase": "调温度", "raw": 0.55, "prob": 0.74, "divergence": True},
                {"phrase": "烧水", "raw": 2.05, "prob": 0.71, "rescued": True},
                {"phrase": "关窗", "raw": 0.35, "prob": 0.18},
                {"phrase": "拉窗帘", "raw": 0.15, "prob": 0.11},
                {"phrase": "放音乐", "raw": 0.05, "prob": 0.07, "spike_target": True},
            ],
        },
        {
            "id": "pure_negative",
            "title": "纯负样本：「伊讲开心已经归到旧档里。」（硬负样本）",
            "expected": [],
            "note": "整句没有任何已注册目标词，靠 min_probability 门限实现完整负样本拒识，压低 FAR。",
            "candidates": [
                {"phrase": "开灯", "raw": 0.82, "prob": 0.34},
                {"phrase": "关窗", "raw": 0.51, "prob": 0.19},
                {"phrase": "烧水", "raw": 0.38, "prob": 0.14},
                {"phrase": "调温度", "raw": 0.24, "prob": 0.11},
                {"phrase": "开风扇", "raw": 0.12, "prob": 0.08},
                {"phrase": "放音乐", "raw": 0.19, "prob": 0.10, "spike_target": True},
            ],
        },
    ]

    return {
        "calibrator_params": {
            "weights": _round_list(cal_bal.weights, 4),
            "mean": _round_list(cal_bal._mean, 4),
            "scale": _round_list(cal_bal._scale, 4),
            "log_prior_offset": round(float(cal_bal.log_prior_offset), 4),
            "w1_over_w0": round(float(np.exp(cal_bal.log_prior_offset)), 3),
            "positive_rate": round(float(np.mean(y_eval)), 4),
        },
        "reliability": {
            "uncorrected": {
                "ece": round(expected_calibration_error(probs_uncorrected, y_eval), 4),
                "brier": round(brier_score(probs_uncorrected, y_eval), 4),
                "bins": _reliability_bins(probs_uncorrected, y_eval),
            },
            "corrected": {
                "ece": round(expected_calibration_error(probs_corrected, y_eval), 4),
                "brier": round(brier_score(probs_corrected, y_eval), 4),
                "bins": _reliability_bins(probs_corrected, y_eval),
            },
            "plain_unweighted": {
                "ece": round(expected_calibration_error(probs_plain, y_eval), 4),
                "brier": round(brier_score(probs_plain, y_eval), 4),
                "bins": _reliability_bins(probs_plain, y_eval),
            },
        },
        "operating_surface": [
            {
                "cap": int(row["cap"]),
                "margin": float(row["margin"]),
                "min_probability": float(row["min_probability"]),
                "f1": round(float(row["f1"]), 4),
                "far": round(float(row["false_alarm_rate"]), 4),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["occurrence_recall"]), 4),
            }
            for row in surface
        ],
        "utterances": utterances,
    }


def _make_eval_lie_detector_data() -> dict[str, Any]:
    """Multi-span Hungarian matching showcase + Head/Body/Tail frequency_bias presets."""
    timeline = {
        "utterance": "先去开灯，再烧水，顺便拉窗帘，最后打开电视。",
        "duration": 5.0,
        "iou_threshold": 0.5,
        "expected_spans": [
            {"id": "e0", "phrase": "开灯", "start": 0.45, "end": 1.10},
            {"id": "e1", "phrase": "烧水", "start": 1.55, "end": 2.20},
            {"id": "e2", "phrase": "拉窗帘", "start": 2.75, "end": 3.55},
            {"id": "e3", "phrase": "打开电视", "start": 4.00, "end": 4.80},
        ],
        "predicted_spans": [
            {"id": "p0", "phrase": "开灯", "start": 0.52, "end": 1.16, "char_source": "reading"},
            {"id": "p1", "phrase": "烧水", "start": 1.62, "end": 2.28, "char_source": "reading"},
            {"id": "p2", "phrase": "拉窗帘", "start": 2.88, "end": 3.68, "char_source": "text-fallback"},
            {"id": "p3", "phrase": "打开电视", "start": 3.92, "end": 4.70, "char_source": "reading"},
            {"id": "p4", "phrase": "关窗", "start": None, "end": None, "char_source": "no-position"},
        ],
    }

    py_check = evaluate_multi_span_localisation(
        [
            {
                "sample_id": "demo-long-01",
                "expected_spans": timeline["expected_spans"],
                "predicted_spans": timeline["predicted_spans"],
            }
        ],
        iou_threshold=0.5,
        strict_contract=True,
    )

    # Frequency bias presets (Head / Body / Tail)
    lex_counts = {
        "开灯": 140,
        "关窗": 115,
        "烧水": 92,
        "调温度": 48,
        "开风扇": 39,
        "拉窗帘": 31,
        "放音乐": 14,
        "查日程": 11,
        "扫地": 7,
        "打开电视": 5,
    }
    freq_classes = {e.phrase: e.frequency_class for e in demo_lexicon()}

    head_biased_recalls = {
        "开灯": 0.96,
        "关窗": 0.94,
        "烧水": 0.91,
        "调温度": 0.78,
        "开风扇": 0.72,
        "拉窗帘": 0.66,
        "放音乐": 0.42,
        "查日程": 0.36,
        "扫地": 0.28,
        "打开电视": 0.20,
    }
    balanced_recalls = {
        "开灯": 0.88,
        "关窗": 0.87,
        "烧水": 0.85,
        "调温度": 0.84,
        "开风扇": 0.83,
        "拉窗帘": 0.85,
        "放音乐": 0.82,
        "查日程": 0.80,
        "扫地": 0.81,
        "打开电视": 0.79,
    }

    def _pack_freq(recalls: dict[str, float]) -> dict[str, Any]:
        fb = frequency_bias(recalls, lex_counts)
        total_occ = sum(lex_counts.values())
        micro = sum(recalls[w] * lex_counts[w] for w in lex_counts) / total_occ
        macro = float(np.mean(list(recalls.values())))
        tail_words = [w for w, c in freq_classes.items() if c == "tail"]
        tail_macro = float(np.mean([recalls[w] for w in tail_words]))
        return {
            "micro_recall": round(micro, 4),
            "macro_recall": round(macro, 4),
            "tail_recall": round(tail_macro, 4),
            "pearson_log": round(fb["pearson_rho_log_frequency"], 4),
            "spearman": round(fb["spearman_rho_log_frequency"], 4),
            "points": [
                {
                    "word": w,
                    "count": lex_counts[w],
                    "recall": recalls[w],
                    "freq_class": freq_classes[w],
                }
                for w in WORDS
            ],
        }

    return {
        "timeline": timeline,
        "py_multi_span_reference": py_check,
        "floor_trap_demo": {
            "positive_records": 50,
            "negative_records": 50,
            "positive_complete_hits": 18,
            "naive_sentence_recall_complete": round((18 + 50) / 100, 4),
            "v3_sentence_recall_complete": round(18 / 50, 4),
        },
        "frequency_presets": {
            "head_biased": _pack_freq(head_biased_recalls),
            "balanced": _pack_freq(balanced_recalls),
        },
    }


def _make_manifest_sandbox_data() -> dict[str, Any]:
    """Run real build_benchmark_plan and validate_plan on default lexicon."""
    lexicon = demo_lexicon()
    plan = demo_plan("small", multi_phrase_ratio=0.3)
    summary = validate_plan(plan, lexicon=lexicon)
    return {
        "default_words": list(WORDS),
        "default_confusions": {t: c for t, c in CONFUSIONS},
        "default_backgrounds": list(BACKGROUNDS),
        "demo_validation": summary,
    }


def main() -> None:
    bundle = {
        "version": "v3.0",
        "dsp_presets": _make_dsp_presets(),
        "calibration_decoder": _make_calibration_and_decoder_data(),
        "eval_lie_detector": _make_eval_lie_detector_data(),
        "manifest_sandbox": _make_manifest_sandbox_data(),
    }
    out_path = ROOT / "studio" / "data" / "synthbench_bundle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Exported {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
