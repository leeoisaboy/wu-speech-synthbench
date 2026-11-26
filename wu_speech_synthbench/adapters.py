from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .synth_pipeline.manifest import SynthesisRecord


@dataclass(frozen=True)
class SynthesizedRecord:
    """Intermediate manifest record bridging ``synth_pipeline`` and ``auto_timestamp``.

    Produced by a ``SynthesisAdapter`` after rendering a ``SynthesisRecord`` to audio.
    """

    sample_id: str
    subset: str
    text: str
    audio_path: str
    sample_rate: int
    duration_seconds: float
    carrier_spans: dict[str, list[int]]
    expected_targets: list[str]
    anchors: dict[str, list[float]] = field(default_factory=dict)
    provider: str | None = None
    voice_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SynthesizedRecord":
        known = {
            "sample_id",
            "subset",
            "text",
            "audio_path",
            "sample_rate",
            "duration_seconds",
            "carrier_spans",
            "char_spans",
            "expected_targets",
            "expected_phrases",
            "anchors",
            "provider",
            "voice_id",
            "extra",
        }
        data = dict(row)
        extra = dict(data.get("extra") or {})
        for key, value in data.items():
            if key not in known:
                extra[key] = value
        return cls(
            sample_id=str(data["sample_id"]),
            subset=str(data["subset"]),
            text=str(data["text"]),
            audio_path=str(data["audio_path"]),
            sample_rate=int(data["sample_rate"]),
            duration_seconds=float(data["duration_seconds"]),
            carrier_spans={
                str(k): [int(v[0]), int(v[1])]
                for k, v in (data.get("carrier_spans") or data.get("char_spans") or {}).items()
            },
            expected_targets=list(data.get("expected_targets") or data.get("expected_phrases") or []),
            anchors={str(k): [float(t) for t in v] for k, v in (data.get("anchors") or {}).items()},
            provider=data.get("provider"),
            voice_id=data.get("voice_id"),
            extra=extra,
        )


@dataclass(frozen=True)
class RetrievalResult:
    """Standardized retrieval model output bridging external models to ``set_prediction`` / ``eval_metrics``."""

    sample_id: str
    expected_targets: list[str]
    predicted_phrases: list[str]
    scores: list[dict[str, Any]] = field(default_factory=list)
    predicted_spans: list[dict[str, Any]] = field(default_factory=list)
    subset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload["expected_phrases"] = list(self.expected_targets)
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
        return payload


@runtime_checkable
class SynthesisAdapter(Protocol):
    """Protocol for external TTS / dialect speech synthesis backends."""

    provider: str

    def synthesize(
        self,
        record: SynthesisRecord,
        output_dir: Path,
        voice_id: str | None = None,
    ) -> SynthesizedRecord:
        """Synthesize a single ``SynthesisRecord`` and return its intermediate manifest entry."""
        ...

    def synthesize_batch(
        self,
        records: Sequence[SynthesisRecord],
        output_dir: Path,
        voice_id: str | None = None,
    ) -> list[SynthesizedRecord]:
        """Synthesize a batch of records."""
        ...


@runtime_checkable
class RetrievalAdapter(Protocol):
    """Protocol for external phrase retrieval / KWS models."""

    model_name: str

    def score_utterance(
        self,
        record: SynthesizedRecord,
        lexicon_phrases: Sequence[str],
    ) -> RetrievalResult:
        """Score enrolled phrases against a synthesized utterance."""
        ...


class MockSynthesisAdapter:
    """Deterministic zero-dependency adapter for local end-to-end pipeline dry runs.

    Generates synthetic metadata and anchor timestamps without requiring external TTS weights.
    """

    def __init__(self, provider: str = "mock-synth", sample_rate: int = 16000, char_duration_s: float = 0.14):
        self.provider = provider
        self.sample_rate = int(sample_rate)
        self.char_duration_s = float(char_duration_s)

    def synthesize(
        self,
        record: SynthesisRecord,
        output_dir: Path,
        voice_id: str | None = None,
    ) -> SynthesizedRecord:
        n_chars = max(len(record.text), 1)
        duration = round(n_chars * self.char_duration_s + 0.20, 4)
        anchors: dict[str, list[float]] = {}
        for phrase, (start_idx, end_idx) in record.carrier_spans.items():
            anchors[phrase] = [
                round(0.10 + (char_idx + 0.5) * self.char_duration_s, 4)
                for char_idx in range(start_idx, end_idx)
            ]
        audio_path = str(Path(output_dir) / f"{record.sample_id}.wav")
        return SynthesizedRecord(
            sample_id=record.sample_id,
            subset=record.subset,
            text=record.text,
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            carrier_spans={k: list(v) for k, v in record.carrier_spans.items()},
            expected_targets=list(record.expected_targets),
            anchors=anchors,
            provider=self.provider,
            voice_id=voice_id or "mock-voice-0",
            extra={"position": record.position, "mock": True},
        )

    def synthesize_batch(
        self,
        records: Sequence[SynthesisRecord],
        output_dir: Path,
        voice_id: str | None = None,
    ) -> list[SynthesizedRecord]:
        return [self.synthesize(record, output_dir, voice_id=voice_id) for record in records]


def write_synthesized_manifest(records: Sequence[SynthesizedRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def load_synthesized_manifest(path: str | Path) -> list[SynthesizedRecord]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return [SynthesizedRecord.from_dict(json.loads(line)) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Quartet & External Engine Schema Adapters (v4.0)
# ---------------------------------------------------------------------------

def from_disentangle_matches(
    sample_id: str,
    matches: Sequence[Mapping[str, Any]],
    *,
    expected_targets: Sequence[str] = (),
    frame_rate_hz: float = 50.0,
    subset: str | None = None,
) -> RetrievalResult:
    """Convert ``disentangle-acoustic-print`` match dicts into a standard ``RetrievalResult``."""
    score_by_key: dict[str, float] = {}
    pred_spans: list[dict[str, Any]] = []
    for item in matches:
        key = str(item.get("key") or item.get("phrase") or "")
        sc = float(item.get("score", 0.0))
        if not key:
            continue
        if key not in score_by_key or sc > score_by_key[key]:
            score_by_key[key] = sc
        start = item.get("start")
        end = item.get("end")
        if start is not None and end is not None:
            pred_spans.append(
                {
                    "phrase": key,
                    "start": float(start) / max(frame_rate_hz, 1.0),
                    "end": float(end) / max(frame_rate_hz, 1.0),
                    "char_source": "disentangle-acoustic-print",
                }
            )
    ordered = sorted(score_by_key.items(), key=lambda kv: -kv[1])
    return RetrievalResult(
        sample_id=str(sample_id),
        expected_targets=list(expected_targets),
        predicted_phrases=[k for k, _ in ordered],
        scores=[{"phrase": k, "score": s} for k, s in ordered],
        predicted_spans=pred_spans,
        subset=subset or ("positive" if expected_targets else "negative"),
    )


def from_audiollm_kws_record(row: Mapping[str, Any]) -> RetrievalResult:
    """Convert an ``audiollm-latent-kws`` detection record into a standard ``RetrievalResult``."""
    sid = str(row.get("sample_id") or row.get("utt") or "")
    expected = [
        str(g["word"]) for g in (row.get("gold") or []) if isinstance(g, Mapping) and g.get("word")
    ] or list(row.get("expected_targets") or row.get("expected_phrases") or [])
    dets = list(row.get("detections") or row.get("predictions") or [])
    score_by_key: dict[str, float] = {}
    pred_spans: list[dict[str, Any]] = []
    for det in dets:
        key = str(det.get("word") or det.get("phrase") or det.get("key") or "")
        sc = float(det.get("score", 0.0))
        if not key:
            continue
        if key not in score_by_key or sc > score_by_key[key]:
            score_by_key[key] = sc
        t0 = det.get("t0", det.get("start"))
        t1 = det.get("t1", det.get("end"))
        pred_spans.append(
            {
                "phrase": key,
                "start": None if t0 is None else float(t0),
                "end": None if t1 is None else float(t1),
                "char_source": "no-position" if (t0 is None or t1 is None) else "audiollm-latent-kws",
            }
        )
    ordered = sorted(score_by_key.items(), key=lambda kv: -kv[1])
    return RetrievalResult(
        sample_id=sid,
        expected_targets=expected,
        predicted_phrases=[k for k, _ in ordered],
        scores=[{"phrase": k, "score": s} for k, s in ordered],
        predicted_spans=pred_spans,
        subset=str(row.get("subset") or ("positive" if expected else "negative")),
    )


def from_crossmodal_align_record(row: Mapping[str, Any]) -> RetrievalResult:
    """Convert a ``dialect-crossmodal-align`` calibrated score record into a ``RetrievalResult``."""
    sid = str(row.get("sample_id") or row.get("clip_id") or "")
    expected = list(row.get("expected_targets") or ([row["word"]] if row.get("word") else []))
    raw_scores = row.get("scores") or {}
    if isinstance(raw_scores, Mapping):
        items = [(str(k), float(v)) for k, v in raw_scores.items()]
    else:
        items = [(str(x["phrase"]), float(x["score"])) for x in raw_scores]
    items.sort(key=lambda kv: -kv[1])
    return RetrievalResult(
        sample_id=sid,
        expected_targets=expected,
        predicted_phrases=[k for k, _ in items],
        scores=[{"phrase": k, "score": s} for k, s in items],
        predicted_spans=[],
        subset=str(row.get("subset") or ("positive" if expected else "negative")),
    )


def from_asr_transcript(
    sample_id: str,
    transcript: str,
    lexicon_phrases: Sequence[str],
    *,
    expected_targets: Sequence[str] = (),
    word_timestamps: Sequence[Mapping[str, Any]] | None = None,
    token_mode: str = "auto",
    subset: str | None = None,
) -> RetrievalResult:
    """Convert Whisper / FunASR / MFA transcript (+ optional word timestamps) into ``RetrievalResult``."""
    from .synth_pipeline.manifest import phrase_occurs_in_text

    matched: list[str] = []
    pred_spans: list[dict[str, Any]] = []
    ts_list = list(word_timestamps or [])

    for phrase in lexicon_phrases:
        if phrase_occurs_in_text(phrase, transcript, token_mode=token_mode):
            matched.append(phrase)
            span_found = False
            for w in ts_list:
                tok = str(w.get("word") or w.get("text") or "")
                if tok and phrase_occurs_in_text(phrase, tok, token_mode=token_mode):
                    t0 = w.get("start", w.get("t0"))
                    t1 = w.get("end", w.get("t1"))
                    if t0 is not None and t1 is not None:
                        pred_spans.append(
                            {
                                "phrase": phrase,
                                "start": float(t0),
                                "end": float(t1),
                                "char_source": "asr-timestamp",
                            }
                        )
                        span_found = True
                        break
            if not span_found:
                pred_spans.append(
                    {"phrase": phrase, "start": None, "end": None, "char_source": "no-position"}
                )

    return RetrievalResult(
        sample_id=str(sample_id),
        expected_targets=list(expected_targets),
        predicted_phrases=matched,
        scores=[{"phrase": p, "score": 1.0 if p in matched else 0.0} for p in lexicon_phrases],
        predicted_spans=pred_spans,
        subset=subset or ("positive" if expected_targets else "negative"),
    )
