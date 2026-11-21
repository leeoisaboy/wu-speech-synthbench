from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import numpy as np

from wu_speech_synthbench import (
    BenchmarkPlan,
    ConfusionPair,
    LexiconEntry,
    LogisticCalibrator,
    MockSynthesisAdapter,
    SynthesisAdapter,
    SynthesisRecord,
    SynthesizedRecord,
    acoustic_features,
    brier_score,
    build_benchmark_plan,
    decode_set,
    derive_position,
    evaluate_localisation,
    evaluate_multi_span_localisation,
    evaluate_ranking,
    expected_calibration_error,
    extract_candidate_features,
    find_all_occurrences,
    find_substring_conflicts,
    fit_set_calibrator,
    frequency_bias,
    iou,
    load_confusion_pairs,
    load_lexicon,
    load_synthesized_manifest,
    segment_word,
    sweep_set_decisions,
    validate_plan,
    write_synthesized_manifest,
)
from wu_speech_synthbench.cli import main


class WuSpeechSynthBenchV3Tests(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. Corpus Planning (manifest.py)
    # ------------------------------------------------------------------
    def test_modulo3_lexicon_covers_all_positions_for_every_word(self) -> None:
        lexicon = [LexiconEntry("开灯"), LexiconEntry("关窗"), LexiconEntry("烧水")]
        plan = build_benchmark_plan(lexicon, target_count=9, hard_negative_count=9, background_count=6)
        report = validate_plan(plan, lexicon=lexicon)
        self.assertTrue(report["valid"])

        pos_by_word: dict[str, set[str]] = {entry.phrase: set() for entry in lexicon}
        for row in plan.records:
            if row.subset == "positive":
                pos_by_word[row.target_phrase].add(row.position)
        for phrase, positions in pos_by_word.items():
            self.assertEqual(positions, {"head", "middle", "tail"}, f"{phrase} locked to {positions}")

    def test_default_hard_negatives_never_leak_any_enrolled_phrase(self) -> None:
        lexicon = [LexiconEntry("开灯"), LexiconEntry("关窗"), LexiconEntry("烧水")]
        enrolled = {e.phrase for e in lexicon}
        plan = build_benchmark_plan(
            lexicon,
            confusion_pairs=None,
            target_count=3,
            hard_negative_count=9,
            background_count=3,
        )
        report = validate_plan(plan, lexicon=lexicon)
        self.assertTrue(report["valid"])
        self.assertEqual(report["hard_negative_leakages"], [])
        for row in plan.records:
            if row.subset == "hard_negative":
                self.assertEqual(row.expected_targets, [])
                for phrase in enrolled:
                    self.assertNotIn(phrase, row.text)

        # Explicitly leaking another enrolled word in a hard negative must be caught by validate_plan
        bad_plan = build_benchmark_plan(
            lexicon,
            confusion_pairs=[ConfusionPair("开灯", "关窗")],
            target_count=3,
            hard_negative_count=1,
            background_count=0,
        )
        with self.assertRaises(ValueError):
            validate_plan(bad_plan, lexicon=lexicon, strict_negative_leakage=True)
        warn_report = validate_plan(bad_plan, lexicon=lexicon, strict_negative_leakage=False)
        self.assertEqual(len(warn_report["hard_negative_leakages"]), 1)

    def test_hard_negative_position_alignment_and_single_word_fallback(self) -> None:
        plan = build_benchmark_plan(
            [LexiconEntry("开灯")],
            confusion_pairs=None,
            target_count=3,
            hard_negative_count=3,
            background_count=3,
        )
        validate_plan(plan, lexicon=[LexiconEntry("开灯")])
        hn_rows = [row for row in plan.records if row.subset == "hard_negative"]
        self.assertEqual(len(hn_rows), 3)
        for row in hn_rows:
            self.assertEqual(row.expected_targets, [])
            self.assertNotIn(row.target_phrase, row.text)
            assert row.confusion_phrase is not None
            start, end = row.carrier_spans[row.confusion_phrase]
            if row.position == "head":
                self.assertEqual(start, 0)
            elif row.position == "middle":
                self.assertGreater(start, 0)
                self.assertLess(end, len(row.text) - 1)
            elif row.position == "tail":
                self.assertEqual(end, len(row.text) - 1)

    def test_multi_phrase_long_utterance_generation(self) -> None:
        lexicon = [LexiconEntry(w) for w in ("开灯", "关窗", "烧水", "扫地", "查日程")]
        plan = build_benchmark_plan(
            lexicon,
            target_count=10,
            hard_negative_count=3,
            background_count=2,
            multi_phrase_ratio=0.5,
            seed=42,
        )
        report = validate_plan(plan, lexicon=lexicon)
        self.assertTrue(report["valid"])
        multi_rows = [row for row in plan.records if row.subset == "positive" and len(row.expected_targets) >= 2]
        self.assertEqual(len(multi_rows), 5)

    def test_validate_plan_detects_background_leakage_and_substring_conflicts(self) -> None:
        lexicon = [LexiconEntry("开"), LexiconEntry("开灯")]
        self.assertEqual(find_substring_conflicts([e.phrase for e in lexicon]), [{"substring": "开", "superstring": "开灯"}])
        leaking_plan = build_benchmark_plan(
            [LexiconEntry("开灯")],
            background_phrases=["开灯维修"],
            target_count=1,
            hard_negative_count=0,
            background_count=1,
        )
        with self.assertRaises(ValueError):
            validate_plan(leaking_plan, lexicon=[LexiconEntry("开灯")], strict_background_leakage=True)

    # ------------------------------------------------------------------
    # 2. Acoustic Energy-Valley Segmentation (v3 signal-processing fixes)
    # ------------------------------------------------------------------
    def test_parseval_high_band_conservation(self) -> None:
        sample_rate = 16000
        t = np.arange(int(sample_rate * 0.4)) / sample_rate
        # Pure 2600 Hz sine lies strictly inside both speech band (300-3400) and high band (2000-4000)
        tone = np.sin(2.0 * np.pi * 2600.0 * t)
        feats = acoustic_features(tone, sample_rate, bandpass=(300.0, 3400.0), high_band_hz=(2000.0, 4000.0))
        mid = len(feats["energy_db"]) // 2
        # Under Parseval conservation, time-domain speech energy and high-band energy must match closely (< 0.5 dB)
        self.assertAlmostEqual(float(feats["energy_db"][mid]), float(feats["high_band_db"][mid]), delta=0.5)

    def test_word_local_normalization_resists_digital_silence_and_aligns_posteriors(self) -> None:
        sample_rate = 16000
        duration = 1.2
        t = np.arange(int(sample_rate * duration)) / sample_rate
        carrier = np.sin(2.0 * np.pi * 700.0 * t)
        # Surrounding digital silence (0.0 -> -120 dB) + two-char word in [0.40, 0.80] with 10 dB dip at 0.60
        env = np.zeros_like(t)
        word_mask = (t >= 0.40) & (t <= 0.80)
        env[word_mask] = 1.0 - 0.72 * np.exp(-0.5 * ((t[word_mask] - 0.60) / 0.015) ** 2)
        signal = carrier * env

        res = segment_word(signal, sample_rate, "开灯", [0.48, 0.72], flux_weight=0.25)
        self.assertAlmostEqual(res.characters[0].end_time, 0.60, delta=0.025)

        # Posterior crossover placed precisely at 0.60s on a 10ms hop grid (offset 0.0)
        n_post = int(duration / 0.010)
        post = np.zeros((n_post, 2), dtype=np.float64)
        post[:60, 0] = 1.0
        post[60:, 1] = 1.0
        fused = segment_word(
            signal,
            sample_rate,
            "开灯",
            [0.48, 0.72],
            posterior_probs=post,
            posterior_hop_seconds=0.010,
            posterior_offset_seconds=0.0,
            posterior_weight=0.8,
        )
        self.assertEqual(fused.source, "anchors+energy+posterior")
        self.assertAlmostEqual(fused.characters[0].end_time, 0.60, delta=0.02)

    def test_derive_position_multi_occurrence_disambiguation(self) -> None:
        text = "开灯之后别忘了再检查开灯。"
        self.assertEqual(find_all_occurrences(text, "开灯"), [(0, 2), (10, 12)])
        second = derive_position(text, "开灯", expected_char_center=11.0)
        self.assertEqual((second.start_char, second.end_char), (10, 12))
        self.assertEqual(second.occurrence_index, 1)

    # ------------------------------------------------------------------
    # 3. Set Prediction & Calibrator (v3 margin fix, prior correction, ties)
    # ------------------------------------------------------------------
    def test_decode_set_calibrator_does_not_prematurely_break_on_margin(self) -> None:
        class _DummyCalibrator:
            def predict_proba(self, feats: np.ndarray) -> np.ndarray:
                # Deliberately rank index 0 (low z) above index 1 (high z) to test probability-vs-z crossing
                return np.array([0.95, 0.90, 0.10], dtype=np.float64)

        record = {
            "sample_id": "cross-1",
            "expected_phrases": ["b"],
            "scores": [
                {"phrase": "a", "score": -2.0},  # low z -> fails margin
                {"phrase": "b", "score": 3.0},   # max z -> must NOT be cut off by 'a'!
                {"phrase": "c", "score": -2.0},
            ],
        }
        decision = decode_set(
            record,
            cap=2,
            margin=0.5,
            calibrator=_DummyCalibrator(),  # type: ignore[arg-type]
            min_probability=0.5,
        )
        self.assertEqual(decision.selected, ["b"])

    def test_calibrator_prior_odds_correction_prevents_imbalance_drift(self) -> None:
        rng = np.random.default_rng(7)
        # 5% positive prevalence (20 positives, 380 negatives)
        x_neg = rng.normal(0.0, 1.0, size=(380, 1))
        x_pos = rng.normal(1.2, 1.0, size=(20, 1))
        X = np.vstack([x_neg, x_pos])
        y = np.concatenate([np.zeros(380), np.ones(20)])

        unweighted = LogisticCalibrator(l2=0.5, class_weight=None).fit(X, y)
        bal_corrected = LogisticCalibrator(l2=0.5, class_weight="balanced", correct_prior=True).fit(X, y)
        bal_raw = LogisticCalibrator(l2=0.5, class_weight="balanced", correct_prior=False).fit(X, y)

        mean_p_unw = float(np.mean(unweighted.predict_proba(X)))
        mean_p_corr = float(np.mean(bal_corrected.predict_proba(X)))
        mean_p_raw = float(np.mean(bal_raw.predict_proba(X)))

        # Uncorrected balanced drifts to ~0.50; prior-corrected stays close to true 0.05 prevalence
        self.assertGreater(mean_p_raw, 0.35)
        self.assertLess(abs(mean_p_corr - 0.05), 0.05)
        self.assertLess(abs(mean_p_unw - 0.05), 0.03)

    def test_extract_candidate_features_tie_invariance(self) -> None:
        f1 = extract_candidate_features([1.0, 1.0, 0.0], [0.7, 0.7, -1.4])
        f2 = extract_candidate_features([0.0, 1.0, 1.0], [-1.4, 0.7, 0.7])
        # Tied 1.0 items must get identical rank_ratio (0.25) regardless of input order
        self.assertAlmostEqual(float(f1[0, 3]), float(f1[1, 3]))
        self.assertAlmostEqual(float(f1[0, 3]), float(f2[1, 3]))

    # ------------------------------------------------------------------
    # 4. Ranking & Multi-Span Localisation (v3 metric fixes)
    # ------------------------------------------------------------------
    def test_ranking_sentence_recall_complete_not_inflated_by_negatives_and_tie_safe_pr_auc(self) -> None:
        # Completely failing model: 0% recall on positives, 100% false alarms on negatives
        rows = [
            {
                "subset": "positive",
                "expected_phrases": ["开灯"],
                "predicted_phrases": ["关窗"],
                "scores": [{"phrase": "开灯", "score": 0.5}, {"phrase": "关窗", "score": 0.5}],
            },
            {
                "subset": "hard_negative",
                "expected_phrases": [],
                "predicted_phrases": ["开灯"],
                "scores": [{"phrase": "开灯", "score": 0.5}, {"phrase": "关窗", "score": 0.5}],
            },
        ]
        report = evaluate_ranking(rows, top_k=1)
        self.assertEqual(report["positive_records"], 1)
        self.assertEqual(report["negative_records"], 1)
        self.assertEqual(report["sentence_recall_complete"], 0.0)
        self.assertEqual(report["exact_set_match"], 0.0)
        self.assertEqual(report["false_alarm_rate"], 1.0)

        # Permuting tied scores must yield identical pr_auc
        rows_reversed = [
            {**rows[0], "scores": list(reversed(rows[0]["scores"]))},
            {**rows[1], "scores": list(reversed(rows[1]["scores"]))},
        ]
        self.assertAlmostEqual(report["pr_auc"], evaluate_ranking(rows_reversed, top_k=1)["pr_auc"])

    def test_multi_span_localisation_counts_zero_iou_known_words_and_no_position_in_precision(self) -> None:
        rows = [
            {
                "sample_id": "m-zero-and-nopos",
                "expected_spans": [
                    {"phrase": "开灯", "start": 0.0, "end": 1.0},
                    {"phrase": "关窗", "start": 2.0, "end": 3.0},
                ],
                "predicted_spans": [
                    {"phrase": "开灯", "start": 0.0, "end": 1.0, "char_source": "reading"},      # IoU = 1.0
                    {"phrase": "关窗", "start": 5.0, "end": 6.0, "char_source": "reading"},      # same phrase, IoU = 0.0!
                    {"phrase": "烧水", "start": None, "end": None, "char_source": "no-position"}, # unpositionable FP!
                    {"phrase": "扫地", "start": None, "end": None, "char_source": "no-position"}, # unpositionable FP!
                ],
            }
        ]
        rep = evaluate_multi_span_localisation(rows, iou_threshold=0.5, strict_contract=True)
        # Known-word IoU must average (1.0 + 0.0) / 2 = 0.5 (not 1.0!)
        self.assertEqual(rep["known_word_matched_pair_count"], 2)
        self.assertAlmostEqual(rep["multi_span_known_word_iou"], 0.5)
        # End-to-end precision denominator includes no-position predictions: 1 / 4 = 0.25 (not 1/2!)
        self.assertEqual(rep["total_predicted_span_count"], 4)
        self.assertEqual(rep["positionable_predicted_span_count"], 2)
        self.assertAlmostEqual(rep["precision_at_iou"], 0.25)
        self.assertAlmostEqual(rep["positionable_precision_at_iou"], 0.5)

    # ------------------------------------------------------------------
    # 5. Adapters & Enhanced CLI (v3 demo multi-phrase + CLI calibrator)
    # ------------------------------------------------------------------
    def test_cli_demo_multi_phrase_and_evaluate_calibrator_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_out = Path(tmp) / "demo.json"
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                main(["demo", "--multi-phrase-ratio", "0.5", "--output", str(demo_out)])
            val = json.loads(buf.getvalue())
            self.assertIn("multi_span_localisation", val)
            # With multi_phrase_ratio=0.5 on 12 positives, expected_span_count must exceed 12
            self.assertGreater(val["multi_span_localisation"]["expected_span_count"], 12)
            self.assertTrue(demo_out.exists())

            # Test CLI evaluate --sets with --calibrator-train and --min-probabilities
            train_path = Path(tmp) / "cal.jsonl"
            eval_path = Path(tmp) / "eval.jsonl"
            sample_rows = [
                {
                    "sample_id": "p1",
                    "expected_phrases": ["开灯"],
                    "scores": [{"phrase": "开灯", "score": 3.0}, {"phrase": "关窗", "score": 0.0}],
                },
                {
                    "sample_id": "n1",
                    "expected_phrases": [],
                    "scores": [{"phrase": "开灯", "score": -1.0}, {"phrase": "关窗", "score": -1.2}],
                },
            ]
            content = "\n".join(json.dumps(r, ensure_ascii=False) for r in sample_rows) + "\n"
            train_path.write_text(content, encoding="utf-8")
            eval_path.write_text(content, encoding="utf-8")

            buf2 = StringIO()
            with contextlib.redirect_stdout(buf2):
                main(
                    [
                        "evaluate",
                        "--sets",
                        str(eval_path),
                        "--calibrator-train",
                        str(train_path),
                        "--caps",
                        "1,2",
                        "--margins",
                        "1.5",
                        "--min-probabilities",
                        "0.5",
                        "--normalization",
                        "robust",
                    ]
                )
            ev = json.loads(buf2.getvalue())
            self.assertEqual(len(ev["set_curve"]), 2)
            self.assertEqual(ev["set_curve"][0]["min_probability"], 0.5)
            self.assertEqual(ev["set_curve"][0]["false_alarm_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
