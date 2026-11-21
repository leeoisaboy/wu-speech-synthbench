# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m compileall -q .
python -m pytest -q
node --check studio/static/studio.js
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Guidelines

- Keep every plan generator deterministic unless a seed is explicit.
- Preserve half-open character spans, decoupled `carrier_spans` and `expected_targets`, and honest provenance values.
- Enforce `no-position` null-coordinate contracts in localisation tests and evaluators.
- Report per-occurrence, per-word, and frequency-class metrics together.
- Update `docs/DATA_CONTRACT.md` for any schema change.
- Regenerate `studio/data/synthbench_bundle.json` with `scripts/export_synthbench_bundle.py` after changing exported fields.
- Add a focused test for every behavioral change.
- Keep model-specific execution and private data outside the repository.
