from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "studio" / "data" / "synthbench_bundle.json"
STUDIO_JS = ROOT / "studio" / "static" / "studio.js"


class StudioBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    def test_bundle_covers_all_four_studio_modules(self) -> None:
        self.assertEqual(self.bundle["version"], "v3.0")
        self.assertGreaterEqual(len(self.bundle["dsp_presets"]), 2)
        self.assertEqual(len(self.bundle["calibration_decoder"]["utterances"]), 3)
        self.assertEqual(len(self.bundle["eval_lie_detector"]["timeline"]["expected_spans"]), 4)
        self.assertTrue(self.bundle["manifest_sandbox"]["default_words"])

    def test_prior_correction_improves_calibration(self) -> None:
        reliability = self.bundle["calibration_decoder"]["reliability"]
        self.assertLess(reliability["corrected"]["ece"], reliability["uncorrected"]["ece"])
        self.assertLess(reliability["corrected"]["brier"], reliability["uncorrected"]["brier"])

    def test_studio_script_has_no_backend_dependency(self) -> None:
        source = STUDIO_JS.read_text(encoding="utf-8")
        self.assertIn("acousticFeatures", source)
        self.assertIn("normalizeScores", source)
        self.assertIn("computeMetrics", source)
        self.assertIn("buildRecords", source)
        self.assertNotIn("fetch('/api", source)
        self.assertNotIn("XMLHttpRequest", source)
        self.assertIn("IntersectionObserver", source)
        self.assertIn("registerAmbientDrawer", source)
        css = (ROOT / "studio" / "static" / "studio.css").read_text(encoding="utf-8")
        self.assertIn("borderless", css.lower())


if __name__ == "__main__":
    unittest.main()
