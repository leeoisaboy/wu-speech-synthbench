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
