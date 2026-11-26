# Module Map (v4)

| Module | Responsibility | v4 Universal Upgrade Focus |
| --- | --- | --- |
| `adapters.py` | `SynthesisAdapter` / `RetrievalAdapter` protocols, `SynthesizedRecord`, `RetrievalResult`, Quartet & ASR converters | `from_disentangle_matches`, `from_audiollm_kws_record`, `from_crossmodal_align_record`, `from_asr_transcript` |
| `synth_pipeline/manifest.py` | Tri-position & multi-phrase carriers, decoupled `carrier_spans` / `expected_targets`, `validate_plan`, `audit_manifest` | Language-aware `token_mode="auto" \| "cjk" \| "word"` (0% English word-boundary FP) + real-corpus `audit_manifest()` |
| `synth_pipeline/demo.py` | Deterministic `head`/`body`/`tail` smart-home lexicon and small/full plans | Multi-phrase & stratified demo |
| `auto_timestamp/energy_valley.py` | Bandpass + Hann/Parseval-subband/flux framing, adaptive noise floor, word-local valley & posterior boundaries | Dynamic Nyquist clamping (8–48 kHz), quantile noise-floor guard, `tts_clean` / `real_noisy` presets |
| `auto_timestamp/derive_positions.py` | Reading, text fallback, no-position provenance, multi-occurrence disambiguation | Honest label audit on long utterances |
| `set_prediction/multi_kw_decoder.py` | Robust MAD / Z-score set decoding, calibrated rejection, F1/FAR sweep | Default CLI `robust` normalization, calibrator-safe margin `continue`, tie-invariant rank features |
| `set_prediction/score_vector_baseline.py` | Armijo line-search ridge IRLS calibrator, `class_weight`, ECE & Brier score | Exact log-odds prior correction (`correct_prior=True`) verified on real retrieval scores |
| `eval_metrics/ranking.py` | Micro/macro, stratified `head`/`body`/`tail`, MRR/MAP/NDCG, FAR/PR-AUC, Pearson & Spearman bias | Positive-only `sentence_recall_complete`, tie-invariant `pr_auc`, real ASR vs Acoustic frequency bias |
| `eval_metrics/localisation.py` | Contract-enforced single-span & Hungarian multi-span IoU / F1@IoU | Zero-IoU same-phrase inclusion & `no-position`-inclusive precision denominator |
| `benchmarks/verify_release_gates.py` | Automated 4-gate release verifier on real public corpora reports | Pass/Fail & honest capability boundary verification |
| `cli.py` | Reproducible CLI for `demo`, `plan`, `validate`, `audit`, and `evaluate` | Added `audit` subcommand, `--token-mode`, default `--normalization robust` |

## Studio modules

| Module | Responsibility |
| --- | --- |
| `studio/index.html` | Static four-module SynthBench Studio page and quartet navigation |
| `studio/static/studio.css` | Light keynote-style layout, responsive controls, charts, and reading states |
| `studio/static/studio.js` | Browser FFT, acoustic features, decoder recomputation, calibration charts, span matching, and corpus-plan audit |
| `studio/data/synthbench_bundle.json` | Deterministic Python-exported data consumed by the static page |
| `scripts/export_synthbench_bundle.py` | Runs the real NumPy modules and writes the Studio bundle |
