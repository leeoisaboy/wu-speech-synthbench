from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import numpy as np

from wu_speech_synthbench import (
    SEGMENT_PRESETS,
    acoustic_features,
    audit_manifest,
    find_substring_conflicts,
    from_asr_transcript,
    from_audiollm_kws_record,
    from_crossmodal_align_record,
    from_disentangle_matches,
    phrase_occurs_in_text,
    segment_word,
)
from wu_speech_synthbench.cli import main


class WuSpeechSynthBenchV4Tests(unittest.TestCase):
    def test_dynamic_nyquist_clamping_from_8khz_to_48khz(self) -> None:
        for sr in (8000, 16000, 24000, 48000):
            t = np.arange(int(sr * 0.5)) / sr
            sig = np.sin(2.0 * np.pi * 600.0 * t) + 0.3 * np.sin(2.0 * np.pi * 2500.0 * t)
            feats = acoustic_features(sig, sr)
            lo, hi = feats["effective_high_band_hz"]
            self.assertLessEqual(hi, (sr / 2.0) * 0.95 + 1e-6)
            self.assertLess(lo, hi)
            self.assertTrue(np.all(np.isfinite(feats["fused_db"])))

    def test_real_noisy_preset_tightens_edges_under_background_hiss(self) -> None:
        sr = 16000
        rng = np.random.default_rng(7)
        t = np.arange(int(sr * 1.2)) / sr
        env = np.where((t >= 0.40) & (t <= 0.80), 1.0, 0.0)
        speech = env * (np.sin(2.0 * np.pi * 650.0 * t) + 0.4 * np.sin(2.0 * np.pi * 1300.0 * t))
        # High-frequency hiss floor (~12 dB SNR)
        hiss = 0.20 * rng.normal(size=len(t))
        sig = speech + hiss

        clean_mode = segment_word(sig, sr, "开灯", [0.50, 0.70], preset="tts_clean")
        noisy_mode = segment_word(sig, sr, "开灯", [0.50, 0.70], preset="real_noisy")
        self.assertIn("real_noisy", SEGMENT_PRESETS)
        # real_noisy should have tighter (smaller or equal) overshoot than tts_clean under hiss
        err_clean = abs(clean_mode.start_time - 0.40) + abs(clean_mode.end_time - 0.80)
        err_noisy = abs(noisy_mode.start_time - 0.40) + abs(noisy_mode.end_time - 0.80)
        self.assertLessEqual(err_noisy, err_clean + 0.02)

    def test_multilingual_token_mode_zero_english_false_positives_and_full_detection(self) -> None:
        # English word boundary: "in" must NOT match "window" or "morning", MUST match "live in town"
        self.assertFalse(phrase_occurs_in_text("in", "open the window this morning", token_mode="auto"))
        self.assertTrue(phrase_occurs_in_text("in", "they live in town", token_mode="auto"))
        self.assertTrue(phrase_occurs_in_text("IN", "They live in town!", token_mode="auto"))
        # CJK literal substring: "上海" matches "上海话"
        self.assertTrue(phrase_occurs_in_text("上海", "阿拉讲上海话", token_mode="auto"))
        # Conflict detector: English "in" vs "window" has 0 conflicts; "in" vs "in town" has 1 conflict
        self.assertEqual(find_substring_conflicts(["in", "window", "morning"], token_mode="auto"), [])
        self.assertEqual(
            find_substring_conflicts(["in", "in town"], token_mode="auto"),
            [{"substring": "in", "superstring": "in town"}],
        )

    def test_audit_manifest_and_cli_audit_on_bilingual_records(self) -> None:
        rows = [
            {"utt": "en-pos-1", "text": "please turn on the light", "gold": [{"word": "in"}]},
            {"utt": "en-neg-1", "text": "open the window this morning", "gold": []},
            {"utt": "en-neg-2", "text": "put it in the box", "gold": []},
            {"utt": "zh-neg-1", "text": "今朝去上海话剧团", "gold": []},
        ]
        rep = audit_manifest(rows, lexicon=["in", "上海"], token_mode="auto")
        self.assertFalse(rep["valid"])
        leaked_ids = {x["sample_id"] for x in rep["negative_leakages"]}
        # en-neg-1 ("window") must NOT be flagged; en-neg-2 ("in") and zh-neg-1 ("上海") MUST be flagged!
        self.assertEqual(leaked_ids, {"en-neg-2", "zh-neg-1"})

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.jsonl"
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                main(["audit", "--manifest", str(p)])
            cli_out = json.loads(buf.getvalue())
            self.assertEqual(cli_out["records"], 4)

    def test_quartet_and_asr_schema_adapters(self) -> None:
        r1 = from_disentangle_matches(
            "u1",
            [{"key": "开灯", "score": 0.88, "start": 25, "end": 50}],
            expected_targets=["开灯"],
            frame_rate_hz=50.0,
        )
        self.assertEqual(r1.predicted_phrases, ["开灯"])
        self.assertAlmostEqual(r1.predicted_spans[0]["start"], 0.5)

        r2 = from_audiollm_kws_record(
            {
                "utt": "u2",
                "gold": [{"word": "烧水"}],
                "detections": [{"word": "烧水", "score": 0.91, "t0": 1.0, "t1": 1.4}],
            }
        )
        self.assertEqual(r2.expected_targets, ["烧水"])
        self.assertEqual(r2.predicted_spans[0]["char_source"], "audiollm-latent-kws")

        r3 = from_crossmodal_align_record(
            {"clip_id": "c1", "word": "关窗", "scores": {"关窗": 0.92, "开灯": 0.12}}
        )
        self.assertEqual(r3.predicted_phrases[0], "关窗")

        r4 = from_asr_transcript(
            "u4",
            "please open the window now",
            ["in", "window"],
            expected_targets=["window"],
            word_timestamps=[{"word": "window", "start": 1.1, "end": 1.5}],
        )
        self.assertEqual(r4.predicted_phrases, ["window"])
        self.assertAlmostEqual(r4.predicted_spans[0]["start"], 1.1)


if __name__ == "__main__":
    unittest.main()
