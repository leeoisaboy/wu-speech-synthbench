# Module Map (v3)

| Module | Responsibility | v3 Upgrade Focus |
| --- | --- | --- |
| `adapters.py` | `SynthesisAdapter` / `RetrievalAdapter` protocols, `SynthesizedRecord`, `RetrievalResult`, `MockSynthesisAdapter` | End-to-end TTS & retrieval handoff |
| `synth_pipeline/manifest.py` | Tri-position & multi-phrase carriers, decoupled `carrier_spans` / `expected_targets`, validation | Disjoint unregistered fallback confusions & full-lexicon `hard_negative` leakage audit |
| `synth_pipeline/demo.py` | Deterministic `head`/`body`/`tail` smart-home lexicon and small/full plans | Multi-phrase & stratified demo |
| `auto_timestamp/energy_valley.py` | Bandpass + Hann/Parseval-subband/flux framing, rollback edge expansion, word-local valley & posterior boundaries | Word-local Min-Max normalization, Parseval high-band conservation, frame-center posterior alignment |
| `auto_timestamp/derive_positions.py` | Reading, text fallback, no-position provenance, multi-occurrence disambiguation | Honest label audit on long utterances |
| `set_prediction/multi_kw_decoder.py` | Z-score / robust MAD set decoding, calibrated rejection, F1/FAR sweep | Calibrator-safe margin `continue`, `prob_margin`, tie-invariant rank features, `class_weight=None` default |
| `set_prediction/score_vector_baseline.py` | Armijo line-search ridge IRLS calibrator, `class_weight`, ECE & Brier score | Exact log-odds prior correction (`correct_prior=True`) |
| `eval_metrics/ranking.py` | Micro/macro, stratified `head`/`body`/`tail`, MRR/MAP/NDCG, FAR/PR-AUC, Pearson & Spearman bias | Positive-only `sentence_recall_complete` & tie-invariant `pr_auc` |
| `eval_metrics/localisation.py` | Contract-enforced single-span & Hungarian multi-span IoU / F1@IoU | Zero-IoU same-phrase inclusion & `no-position`-inclusive precision denominator |
| `cli.py` | Reproducible CLI for `demo`, `plan`, `validate`, and `evaluate` | Multi-phrase demo flow, out-of-sample calibration split, `--calibrator-train` & `--min-probabilities` |

## Studio modules

| Module | Responsibility |
| --- | --- |
| `studio/index.html` | Static four-module SynthBench Studio page and quartet navigation |
| `studio/static/studio.css` | Light keynote-style layout, responsive controls, charts, and reading states |
| `studio/static/studio.js` | Browser FFT, acoustic features, decoder recomputation, calibration charts, span matching, and corpus-plan audit |
| `studio/data/synthbench_bundle.json` | Deterministic Python-exported data consumed by the static page |
| `scripts/export_synthbench_bundle.py` | Runs the real NumPy modules and writes the Studio bundle |
