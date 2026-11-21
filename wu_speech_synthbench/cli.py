from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .auto_timestamp import derive_position, segment_word
from .eval_metrics import evaluate_localisation, evaluate_multi_span_localisation, evaluate_ranking
from .set_prediction import fit_set_calibrator, sweep_set_decisions
from .synth_pipeline import (
    BenchmarkPlan,
    SynthesisRecord,
    build_benchmark_plan,
    load_background_phrases,
    load_confusion_pairs,
    load_lexicon,
    validate_plan,
)
from .synth_pipeline.demo import demo_lexicon, demo_plan


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_optional_float_list(raw: str) -> list[float | None]:
    output: list[float | None] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        output.append(None if token.lower() == "none" else float(token))
    return output or [None]


def _build_synthetic_eval_slice(
    plan: BenchmarkPlan,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate deterministic ranking, single-span, multi-span, and set records for a plan."""
    rng = np.random.default_rng(seed)
    lexicon = demo_lexicon()
    all_phrases = [entry.phrase for entry in lexicon]
    ranking: list[dict[str, Any]] = []
    localisation: list[dict[str, Any]] = []
    multi_span_loc: list[dict[str, Any]] = []
    sets: list[dict[str, Any]] = []

    for row in plan.records:
        if row.subset == "positive":
            expected_list = list(row.expected_targets)
            expected_set = set(expected_list)
            rivals = [p for p in all_phrases if p not in expected_set]
            pool_phrases = list(expected_list) + rivals[: max(4 - len(expected_list), 2)]
            scores = [
                float(rng.normal(2.2 if phrase in expected_set else 0.1, 0.35))
                for phrase in pool_phrases
            ]
            ranked = [p for _, p in sorted(zip(scores, pool_phrases), key=lambda pair: -pair[0])]
            ranking.append(
                {
                    "sample_id": row.sample_id,
                    "subset": row.subset,
                    "expected_phrases": expected_list,
                    "predicted_phrases": ranked[: max(3, len(expected_list))],
                    "scores": [{"phrase": p, "score": s} for p, s in zip(pool_phrases, scores)],
                }
            )
            sets.append(
                {
                    "sample_id": row.sample_id,
                    "subset": row.subset,
                    "expected_phrases": expected_list,
                    "scores": [{"phrase": p, "score": s} for p, s in zip(pool_phrases, scores)],
                }
            )
            # Single-span localisation uses the primary target
            primary = expected_list[0]
            span = row.carrier_spans[primary]
            selected = ranked[0]
            derived = derive_position(row.text, selected)
            localisation.append(
                {
                    "sample_id": row.sample_id,
                    "expected_phrase": primary,
                    "expected_start": float(span[0]),
                    "expected_end": float(span[1]),
                    "predicted_phrase": selected,
                    "predicted_start": float(derived.start_char) if derived.start_char is not None else None,
                    "predicted_end": float(derived.end_char) if derived.end_char is not None else None,
                    "char_source": derived.source.value,
                }
            )
            # Multi-span localisation evaluates all expected targets in the utterance
            exp_spans = [
                {"phrase": p, "start": float(row.carrier_spans[p][0]), "end": float(row.carrier_spans[p][1])}
                for p in expected_list
            ]
            pred_spans: list[dict[str, Any]] = []
            for pred_phrase in ranked[: len(expected_list)]:
                d_pos = derive_position(row.text, pred_phrase)
                pred_spans.append(
                    {
                        "phrase": pred_phrase,
                        "start": float(d_pos.start_char) if d_pos.start_char is not None else None,
                        "end": float(d_pos.end_char) if d_pos.end_char is not None else None,
                        "char_source": d_pos.source.value,
                    }
                )
            multi_span_loc.append(
                {
                    "sample_id": row.sample_id,
                    "expected_spans": exp_spans,
                    "predicted_spans": pred_spans,
                }
            )
        else:
            pool_phrases = all_phrases[:4]
            scores = [float(rng.normal(-0.4, 0.35)) for _ in pool_phrases]
            ranking.append(
                {
                    "sample_id": row.sample_id,
                    "subset": row.subset,
                    "expected_phrases": [],
                    "predicted_phrases": [],
                    "scores": [{"phrase": p, "score": s} for p, s in zip(pool_phrases, scores)],
                }
            )
            sets.append(
                {
                    "sample_id": row.sample_id,
                    "subset": row.subset,
                    "expected_phrases": [],
                    "scores": [{"phrase": p, "score": s} for p, s in zip(pool_phrases, scores)],
                }
            )
    return ranking, localisation, multi_span_loc, sets


def _timestamp_demo() -> dict[str, Any]:
    sample_rate = 16000
    time_axis = np.arange(int(sample_rate * 0.8)) / sample_rate
    carrier = np.sin(2.0 * np.pi * 600.0 * time_axis) + 0.5 * np.sin(2.0 * np.pi * 1400.0 * time_axis)
    envelope = sum(
        amplitude * np.exp(-0.5 * ((time_axis - center) / 0.045) ** 2)
        for center, amplitude in ((0.16, 1.0), (0.36, 0.9), (0.58, 1.1))
    )
    signal = carrier * envelope + 0.002 * np.random.default_rng(17).normal(size=len(time_axis))
    word = segment_word(signal, sample_rate, "烧水", [0.16, 0.36])
    return {
        "word_span": {"start_time": word.start_time, "end_time": word.end_time, "source": word.source},
        "characters": [span.__dict__ for span in word.characters],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wu-speech-synthbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a small deterministic methodology audit")
    demo.add_argument("--scale", choices=("small", "full"), default="small")
    demo.add_argument("--multi-phrase-ratio", type=float, default=0.0)
    demo.add_argument("--output", type=Path, help="optional path to write JSON report")

    plan = subparsers.add_parser("plan", help="write a synthesis plan in JSON Lines")
    plan.add_argument("--lexicon", type=Path, required=True)
    plan.add_argument("--confusions", type=Path)
    plan.add_argument("--backgrounds", type=Path, help="optional background phrases file (.jsonl/.csv/.txt)")
    plan.add_argument("--templates", type=Path, help="optional JSON file mapping head/middle/tail to templates")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--positive-count", type=int, default=2500)
    plan.add_argument("--hard-negative-count", type=int, default=1500)
    plan.add_argument("--background-count", type=int, default=1000)
    plan.add_argument("--multi-phrase-ratio", type=float, default=0.0)
    plan.add_argument("--seed", type=int, default=17)

    validate = subparsers.add_parser("validate", help="validate a generated JSON Lines plan")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--lexicon", type=Path, help="optional lexicon for negative leakage & substring checks")
    validate.add_argument("--no-strict-leakage", action="store_true", help="warn instead of raising on negative leakage")
    validate.add_argument("--output", type=Path, help="optional path to write JSON validation report")

    evaluate = subparsers.add_parser("evaluate", help="evaluate ranking, localisation, and set-prediction JSON Lines")
    evaluate.add_argument("--ranking", type=Path)
    evaluate.add_argument("--localisation", type=Path)
    evaluate.add_argument("--multi-span-localisation", type=Path)
    evaluate.add_argument("--sets", type=Path)
    evaluate.add_argument("--calibrator-train", type=Path, help="optional labeled sets JSONL to fit LogisticCalibrator")
    evaluate.add_argument("--lexicon", type=Path, help="optional lexicon for head/body/tail stratified recalls")
    evaluate.add_argument("--top-k", type=int, default=1)
    evaluate.add_argument("--caps", type=str, default="1,2,3", help="comma-separated cap grid")
    evaluate.add_argument("--margins", type=str, default="0.5,1.0,1.5,2.0", help="comma-separated margin grid")
    evaluate.add_argument("--minimum-zs", type=str, default="none", help="comma-separated minimum_z grid (use 'none')")
    evaluate.add_argument("--min-probabilities", type=str, default="none", help="comma-separated min_probability grid (use 'none')")
    evaluate.add_argument("--prob-margin", type=float, default=None, help="optional margin in calibrated probability space")
    evaluate.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    evaluate.add_argument("--normalization", choices=("zscore", "robust"), default="zscore")
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.add_argument("--no-strict-contract", action="store_true", help="collect localisation contract violations instead of raising")
    evaluate.add_argument("--output", type=Path, help="optional path to write JSON evaluation report")
    return parser


def _emit(output: dict[str, Any], output_path: Path | None) -> None:
    rendered = _json(output)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        lexicon = demo_lexicon()
        eval_plan = demo_plan(args.scale, multi_phrase_ratio=args.multi_phrase_ratio)
        # Independent calibration split prevents in-sample data leakage in demo
        cal_plan = demo_plan("small", multi_phrase_ratio=args.multi_phrase_ratio)
        _, _, _, cal_sets = _build_synthetic_eval_slice(cal_plan, seed=101)
        ranking, localisation, multi_span_loc, eval_sets = _build_synthetic_eval_slice(eval_plan, seed=17)

        calibrator = fit_set_calibrator(cal_sets, normalization="robust", class_weight=None)
        output = {
            "plan_summary": validate_plan(eval_plan, lexicon=lexicon),
            "timestamp": _timestamp_demo(),
            "ranking": evaluate_ranking(ranking, top_k=1, lexicon=lexicon),
            "localisation": evaluate_localisation(localisation, strict_contract=True),
            "multi_span_localisation": evaluate_multi_span_localisation(multi_span_loc, strict_contract=True),
            "set_curve": sweep_set_decisions(
                eval_sets,
                caps=(1, 2, 3),
                margins=(0.5, 1.5),
                minimum_zs=(None, 0.5),
                normalization="robust",
                calibrator=calibrator,
                min_probabilities=(0.5,),
            ),
        }
        _emit(output, args.output)
    elif args.command == "plan":
        lexicon = load_lexicon(args.lexicon)
        confusions = load_confusion_pairs(args.confusions) if args.confusions else []
        backgrounds = load_background_phrases(args.backgrounds) if args.backgrounds else None
        templates = None
        if args.templates:
            with args.templates.open(encoding="utf-8") as handle:
                templates = json.load(handle)
        plan = build_benchmark_plan(
            lexicon,
            confusions,
            background_phrases=backgrounds,
            target_count=args.positive_count,
            hard_negative_count=args.hard_negative_count,
            background_count=args.background_count,
            templates=templates,
            multi_phrase_ratio=args.multi_phrase_ratio,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan.to_jsonl(), encoding="utf-8")
        output = validate_plan(plan, lexicon=lexicon)
        print(_json(output))
    elif args.command == "validate":
        lexicon = load_lexicon(args.lexicon) if args.lexicon else None
        records = [SynthesisRecord.from_dict(row) for row in _read_jsonl(args.plan)]
        output = validate_plan(
            BenchmarkPlan(records),
            lexicon=lexicon,
            strict_background_leakage=not args.no_strict_leakage,
        )
        _emit(output, args.output)
    else:
        lexicon = load_lexicon(args.lexicon) if args.lexicon else None
        strict_contract = not args.no_strict_contract
        output: dict[str, Any] = {}
        if args.ranking:
            ranking_rows = _read_jsonl(args.ranking)
            for top_k in range(1, args.top_k + 1):
                output[f"ranking_top_{top_k}"] = evaluate_ranking(ranking_rows, top_k, lexicon=lexicon)
        if args.localisation:
            output["localisation"] = evaluate_localisation(
                _read_jsonl(args.localisation),
                strict_contract=strict_contract,
            )
        if args.multi_span_localisation:
            output["multi_span_localisation"] = evaluate_multi_span_localisation(
                _read_jsonl(args.multi_span_localisation),
                iou_threshold=args.iou_threshold,
                strict_contract=strict_contract,
            )
        if args.sets:
            calibrator = None
            if args.calibrator_train:
                cw = None if args.class_weight == "none" else args.class_weight
                calibrator = fit_set_calibrator(
                    _read_jsonl(args.calibrator_train),
                    normalization=args.normalization,
                    class_weight=cw,
                )
            output["set_curve"] = sweep_set_decisions(
                _read_jsonl(args.sets),
                caps=_parse_int_list(args.caps),
                margins=_parse_float_list(args.margins),
                minimum_zs=_parse_optional_float_list(args.minimum_zs),
                normalization=args.normalization,
                calibrator=calibrator,
                min_probabilities=_parse_optional_float_list(args.min_probabilities),
                prob_margin=args.prob_margin,
            )
        _emit(output, args.output)
