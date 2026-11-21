from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POSITIONS = ("head", "middle", "tail")

DEFAULT_TEMPLATE_POOLS: dict[str, tuple[str, ...]] = {
    "head": (
        "{word}，帮我看一下今天的事项。",
        "{word}，等会儿记得再核对一遍。",
        "{word}，阿拉待会还要再确认。",
    ),
    "middle": (
        "伊讲{word}要放在房间角落。",
        "今朝先办{word}再处理别的。",
        "表格里写着{word}需要提前准备。",
    ),
    "tail": (
        "饭后请先整理{word}。",
        "出门之前别忘了{word}。",
        "清单最后一项是{word}。",
    ),
}

# Backwards-compatible single-template view (first carrier per position)
DEFAULT_TEMPLATES: dict[str, str] = {
    pos: pool[0] for pos, pool in DEFAULT_TEMPLATE_POOLS.items()
}

DEFAULT_HARD_NEGATIVE_TEMPLATE_POOLS: dict[str, tuple[str, ...]] = {
    "head": (
        "{word}，伊讲这是旧名册。",
        "{word}，先记在草稿纸上。",
        "{word}，这个条目不用再改。",
    ),
    "middle": (
        "伊讲{word}已经归到旧档里。",
        "说明里把{word}标成了备选。",
        "记录上写着{word}暂不启用。",
    ),
    "tail": (
        "备注栏最后选了{word}。",
        "旧清单里只留着{word}。",
        "这次登记的是{word}。",
    ),
}

DEFAULT_MULTI_PHRASE_TEMPLATES: dict[int, tuple[str, ...]] = {
    2: (
        "{w0}，然后记得{w1}。",
        "先去{w0}，回头再{w1}。",
    ),
    3: (
        "{w0}，接着{w1}，最后再去{w2}。",
        "今朝要先{w0}，顺便{w1}，临走前{w2}。",
    ),
    4: (
        "{w0}，再把{w1}弄好，接着{w2}，最后{w3}。",
        "先{w0}，然后{w1}和{w2}，收尾再{w3}。",
    ),
    5: (
        "{w0}，先顺手{w1}，再去{w2}，中间{w3}，最后{w4}。",
    ),
}

DEFAULT_BACKGROUND_PHRASES: tuple[str, ...] = (
    "天气预报",
    "社区通知",
    "菜场价格",
    "公交班次",
    "旧照片",
    "教室课程",
    "图书目录",
    "交通地图",
)

# Neutral unregistered fallback confusions guaranteed not to collide with default templates
DEFAULT_FALLBACK_CONFUSIONS: tuple[str, ...] = (
    "近义项",
    "对照词",
    "替代项",
    "混淆词",
    "参照词",
    "干扰项",
    "未录项",
    "旁支词",
    "候补项",
    "非目标词",
)

_KNOWN_RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "subset",
        "text",
        "position",
        "target_phrase",
        "expected_targets",
        "expected_phrases",
        "carrier_phrases",
        "carrier_spans",
        "char_spans",
        "confusion_phrase",
        "provider",
        "voice_id",
        "extra",
    }
)


@dataclass(frozen=True)
class LexiconEntry:
    phrase: str
    domain: str = "daily"
    frequency_class: str = "tail"
    reading: str = ""


@dataclass(frozen=True)
class ConfusionPair:
    target: str
    confusion: str
    note: str = ""


@dataclass(frozen=True)
class SynthesisRecord:
    """Model-agnostic synthesis job with decoupled alignment and retrieval fields.

    - ``carrier_spans`` / ``carrier_phrases`` describe the phrases actually spoken
      in ``text`` for TTS and acoustic alignment (including confusion/background).
    - ``expected_targets`` describes the enrolled target phrases expected by a
      retrieval system (strictly ``[]`` for ``hard_negative`` and ``background``).
    - ``extra`` preserves unknown extension fields (e.g., ``audio_path``,
      ``sample_rate``, ``duration_seconds``) appended by external synthesizers.
    """

    sample_id: str
    subset: str
    text: str
    position: str
    target_phrase: str
    expected_targets: list[str]
    carrier_spans: dict[str, list[int]]
    carrier_phrases: list[str] = field(default_factory=list)
    confusion_phrase: str | None = None
    provider: str | None = None
    voice_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.carrier_phrases and self.carrier_spans:
            object.__setattr__(self, "carrier_phrases", list(self.carrier_spans.keys()))

    @property
    def expected_phrases(self) -> list[str]:
        """Alias of ``expected_targets`` for downstream retrieval evaluators."""
        return self.expected_targets

    @property
    def char_spans(self) -> dict[str, list[int]]:
        """Alias of ``carrier_spans`` for backwards-compatible span access."""
        return self.carrier_spans

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload["expected_phrases"] = list(self.expected_targets)
        payload["char_spans"] = {k: list(v) for k, v in self.carrier_spans.items()}
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SynthesisRecord":
        data = dict(row)
        subset = str(data["subset"])
        carrier_spans = dict(data.get("carrier_spans") or data.get("char_spans") or {})
        if "expected_targets" in data:
            expected_targets = list(data["expected_targets"])
        elif "expected_phrases" in data:
            raw_expected = list(data["expected_phrases"])
            expected_targets = (
                []
                if subset in ("hard_negative", "background") and raw_expected == list(carrier_spans.keys())
                else raw_expected
            )
        else:
            expected_targets = []
        carrier_phrases = list(data.get("carrier_phrases") or carrier_spans.keys())
        extra = dict(data.get("extra") or {})
        for key, value in data.items():
            if key not in _KNOWN_RECORD_FIELDS:
                extra[key] = value
        return cls(
            sample_id=str(data["sample_id"]),
            subset=subset,
            text=str(data["text"]),
            position=str(data["position"]),
            target_phrase=str(data["target_phrase"]),
            expected_targets=expected_targets,
            carrier_spans={str(k): [int(v[0]), int(v[1])] for k, v in carrier_spans.items()},
            carrier_phrases=[str(p) for p in carrier_phrases],
            confusion_phrase=data.get("confusion_phrase"),
            provider=data.get("provider"),
            voice_id=data.get("voice_id"),
            extra=extra,
        )


@dataclass(frozen=True)
class BenchmarkPlan:
    records: list[SynthesisRecord]
    target_count: int = 2500
    hard_negative_count: int = 1500
    background_count: int = 1000

    @property
    def size(self) -> int:
        return len(self.records)

    def to_jsonl(self) -> str:
        return "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in self.records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_tabular(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_lexicon(path: str | Path) -> list[LexiconEntry]:
    source = Path(path)
    rows = _read_tabular(source)
    output: list[LexiconEntry] = []
    for row in rows:
        phrase = str(row.get("phrase") or row.get("hotword") or row.get("word") or "").strip()
        if phrase:
            output.append(
                LexiconEntry(
                    phrase=phrase,
                    domain=str(row.get("domain") or "daily"),
                    frequency_class=str(row.get("frequency_class") or "tail"),
                    reading=str(row.get("reading") or ""),
                )
            )
    if not output:
        raise ValueError("lexicon is empty")
    return output


def load_confusion_pairs(path: str | Path) -> list[ConfusionPair]:
    source = Path(path)
    rows = _read_tabular(source)
    output: list[ConfusionPair] = []
    for row in rows:
        target = str(row.get("target") or row.get("target_phrase") or "").strip()
        confusion = str(row.get("confusion") or row.get("confusion_phrase") or "").strip()
        if target and confusion:
            output.append(ConfusionPair(target, confusion, str(row.get("note") or "")))
    return output


def load_background_phrases(path: str | Path) -> list[str]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(source)
        phrases = [
            str(row.get("phrase") or row.get("text") or row.get("topic") or "").strip()
            for row in rows
        ]
    elif suffix == ".csv":
        rows = _read_tabular(source)
        phrases = [
            str(row.get("phrase") or row.get("text") or row.get("topic") or "").strip()
            for row in rows
        ]
    else:
        with source.open(encoding="utf-8") as handle:
            phrases = [line.strip() for line in handle if line.strip()]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        raise ValueError("background phrases file is empty")
    return phrases


def _normalize_template_pools(
    templates: Mapping[str, str | Sequence[str]] | None,
    defaults: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = dict(defaults)
    if templates:
        for key, value in templates.items():
            if isinstance(value, str):
                merged[key] = (value,)
            else:
                pool = tuple(str(item) for item in value if str(item))
                if not pool:
                    raise ValueError(f"template pool for {key!r} is empty")
                merged[key] = pool
    if set(merged) != set(POSITIONS):
        raise ValueError("templates must define head, middle, and tail carriers")
    return merged


def _find_spans(text: str, phrases: Sequence[str]) -> tuple[dict[str, list[int]], list[str]]:
    spans: dict[str, list[int]] = {}
    found: list[str] = []
    for phrase in phrases:
        if not phrase or text.count(phrase) != 1:
            return {}, []
        start = text.find(phrase)
        spans[phrase] = [start, start + len(phrase)]
        found.append(phrase)
    return spans, found


def _record(
    sample_id: str,
    subset: str,
    text: str,
    position: str,
    target_phrase: str,
    expected_targets: Sequence[str],
    carrier_phrases: Sequence[str],
    confusion_phrase: str | None = None,
) -> SynthesisRecord:
    spans, found = _find_spans(text, carrier_phrases)
    if found != list(carrier_phrases):
        raise ValueError(f"carrier phrases missing or repeated in text: {carrier_phrases}")
    return SynthesisRecord(
        sample_id=sample_id,
        subset=subset,
        text=text,
        position=position,
        target_phrase=target_phrase,
        expected_targets=list(expected_targets),
        carrier_spans=spans,
        carrier_phrases=list(carrier_phrases),
        confusion_phrase=confusion_phrase,
    )


def _rotate_lexicon(
    entries: Sequence[LexiconEntry],
    count: int,
    template_pools: Mapping[str, Sequence[str]],
) -> Iterable[tuple[int, LexiconEntry, str, str]]:
    """Yield (index, entry, position, text) without modulo-3 position locking."""
    if not entries:
        raise ValueError("lexicon is empty")
    n_entries = len(entries)
    for index in range(count):
        entry = entries[index % n_entries]
        cycle_idx = index // n_entries
        position = POSITIONS[cycle_idx % len(POSITIONS)]
        pool = template_pools[position]
        template = pool[(cycle_idx // len(POSITIONS)) % len(pool)]
        yield index, entry, position, template.format(word=entry.phrase)


def _is_disjoint_from_lexicon(candidate: str, enrolled_phrases: Sequence[str]) -> bool:
    return not any(phrase in candidate or candidate in phrase for phrase in enrolled_phrases)


def _default_confusion_pairs(lexicon: Sequence[LexiconEntry]) -> list[ConfusionPair]:
    """Generate unregistered fallback confusion pairs disjoint from ALL enrolled phrases.

    In v3, fallback confusions are NEVER drawn from other enrolled lexicon entries so
    that hard-negative utterances remain strictly consistent with ``expected_targets=[]``.
    """
    if not lexicon:
        raise ValueError("lexicon is empty")
    enrolled = [entry.phrase for entry in lexicon if entry.phrase]
    safe_pool = [
        cand for cand in DEFAULT_FALLBACK_CONFUSIONS if _is_disjoint_from_lexicon(cand, enrolled)
    ]

    pairs: list[ConfusionPair] = []
    for idx, entry in enumerate(lexicon):
        if idx < len(safe_pool):
            confusion = safe_pool[idx]
        else:
            # Generate deterministic unregistered token guaranteed disjoint from enrolled phrases
            suffix = idx + 1
            confusion = f"非注册词{suffix}"
            while not _is_disjoint_from_lexicon(confusion, enrolled):
                suffix += 1
                confusion = f"非注册词{suffix}"
        pairs.append(ConfusionPair(entry.phrase, confusion, "auto-fallback-unregistered"))
    return pairs


def _select_multi_phrases(
    lexicon: Sequence[LexiconEntry],
    desired_k: int,
    rng: np.random.Generator,
    anchor_entry: LexiconEntry,
) -> list[str]:
    """Select up to ``desired_k`` mutually non-substring distinct phrases."""
    selected = [anchor_entry.phrase]
    others = [entry.phrase for entry in lexicon if entry.phrase != anchor_entry.phrase]
    if not others or desired_k <= 1:
        return selected
    order = rng.permutation(len(others))
    for idx in order:
        cand = others[int(idx)]
        if any(cand in existing or existing in cand for existing in selected):
            continue
        selected.append(cand)
        if len(selected) >= desired_k:
            break
    return selected


def build_benchmark_plan(
    lexicon: Sequence[LexiconEntry],
    confusion_pairs: Sequence[ConfusionPair] | None = None,
    background_phrases: Sequence[str] | None = None,
    target_count: int = 2500,
    hard_negative_count: int = 1500,
    background_count: int = 1000,
    templates: Mapping[str, str | Sequence[str]] | None = None,
    hard_negative_templates: Mapping[str, str | Sequence[str]] | None = None,
    multi_phrase_ratio: float = 0.0,
    phrase_count_weights: Mapping[int, float] | None = None,
    seed: int = 17,
) -> BenchmarkPlan:
    """Create a reproducible positive, hard-negative, and background plan."""
    if not lexicon:
        raise ValueError("lexicon is empty")
    if not (0.0 <= multi_phrase_ratio <= 1.0):
        raise ValueError("multi_phrase_ratio must be in [0.0, 1.0]")

    pos_pools = _normalize_template_pools(templates, DEFAULT_TEMPLATE_POOLS)
    hn_pools = _normalize_template_pools(hard_negative_templates, DEFAULT_HARD_NEGATIVE_TEMPLATE_POOLS)
    rng = np.random.default_rng(seed)

    weights_map = dict(phrase_count_weights or {2: 0.45, 3: 0.30, 4: 0.15, 5: 0.10})
    valid_ks = [k for k in sorted(weights_map) if 2 <= k <= 5 and weights_map[k] > 0]
    if multi_phrase_ratio > 0.0 and valid_ks:
        probs = np.asarray([weights_map[k] for k in valid_ks], dtype=np.float64)
        probs = probs / probs.sum()
    else:
        valid_ks = []
        probs = np.asarray([], dtype=np.float64)

    multi_count = int(round(target_count * multi_phrase_ratio)) if (valid_ks and len(lexicon) >= 2) else 0
    single_count = target_count - multi_count

    records: list[SynthesisRecord] = []

    # 1. Single-phrase tri-position positive carriers
    for index, entry, position, text in _rotate_lexicon(lexicon, single_count, pos_pools):
        records.append(
            _record(
                sample_id=f"positive-{index:05d}",
                subset="positive",
                text=text,
                position=position,
                target_phrase=entry.phrase,
                expected_targets=[entry.phrase],
                carrier_phrases=[entry.phrase],
            )
        )

    # 2. Optional multi-phrase long-utterance positive carriers (2-5 targets)
    for offset in range(multi_count):
        index = single_count + offset
        anchor_entry = lexicon[offset % len(lexicon)]
        desired_k = int(rng.choice(valid_ks, p=probs))
        chosen = _select_multi_phrases(lexicon, desired_k, rng, anchor_entry)
        if len(chosen) >= 2 and len(chosen) in DEFAULT_MULTI_PHRASE_TEMPLATES:
            pool = DEFAULT_MULTI_PHRASE_TEMPLATES[len(chosen)]
            template = pool[offset % len(pool)]
            text = template.format(**{f"w{i}": w for i, w in enumerate(chosen)})
            position = "multi"
        else:
            pos = POSITIONS[(offset // len(lexicon)) % len(POSITIONS)]
            pool = pos_pools[pos]
            text = pool[(offset // (len(lexicon) * len(POSITIONS))) % len(pool)].format(word=chosen[0])
            position = pos
        records.append(
            _record(
                sample_id=f"positive-{index:05d}",
                subset="positive",
                text=text,
                position=position,
                target_phrase=chosen[0],
                expected_targets=chosen,
                carrier_phrases=chosen,
            )
        )

    # 3. Hard-negative carriers (expected_targets = [], carrier_phrases = [confusion])
    pairs = list(confusion_pairs) if confusion_pairs else _default_confusion_pairs(lexicon)
    n_pairs = len(pairs)
    for index in range(hard_negative_count):
        pair = pairs[index % n_pairs]
        if pair.target == pair.confusion or pair.target in pair.confusion:
            raise ValueError(f"invalid confusion pair where target appears in confusion: {pair}")
        cycle_idx = index // n_pairs
        position = POSITIONS[cycle_idx % len(POSITIONS)]
        pool = hn_pools[position]
        text = pool[(cycle_idx // len(POSITIONS)) % len(pool)].format(word=pair.confusion)
        records.append(
            _record(
                sample_id=f"hard-negative-{index:05d}",
                subset="hard_negative",
                text=text,
                position=position,
                target_phrase=pair.target,
                expected_targets=[],
                carrier_phrases=[pair.confusion],
                confusion_phrase=pair.confusion,
            )
        )

    # 4. Background carriers (expected_targets = [], carrier_phrases = [bg_phrase])
    bg_list = list(background_phrases) if background_phrases else list(DEFAULT_BACKGROUND_PHRASES)
    n_bg = len(bg_list)
    for index in range(background_count):
        phrase = bg_list[index % n_bg]
        cycle_idx = index // n_bg
        position = POSITIONS[cycle_idx % len(POSITIONS)]
        pool = pos_pools[position]
        text = pool[(cycle_idx // len(POSITIONS)) % len(pool)].format(word=phrase)
        records.append(
            _record(
                sample_id=f"background-{index:05d}",
                subset="background",
                text=text,
                position=position,
                target_phrase="",
                expected_targets=[],
                carrier_phrases=[phrase],
            )
        )

    return BenchmarkPlan(records, target_count, hard_negative_count, background_count)


def find_substring_conflicts(phrases: Sequence[str]) -> list[dict[str, str]]:
    """Detect pairs of distinct lexicon phrases where one is a substring of another."""
    unique = sorted({p for p in phrases if p})
    conflicts: list[dict[str, str]] = []
    for i, shorter in enumerate(unique):
        for j, longer in enumerate(unique):
            if i != j and shorter in longer:
                conflicts.append({"substring": shorter, "superstring": longer})
    return conflicts


def validate_plan(
    plan: BenchmarkPlan,
    lexicon: Sequence[LexiconEntry] | None = None,
    strict_background_leakage: bool = True,
    strict_negative_leakage: bool | None = None,
) -> dict[str, Any]:
    """Validate spans, subset isolation, full-lexicon negative leakage, and substring conflicts.

    In v3, BOTH ``hard_negative`` and ``background`` subsets are audited against the
    entire ``enrolled_phrases`` set so no registered hotword leaks into a negative utterance.
    """
    enforce_neg_leakage = strict_background_leakage if strict_negative_leakage is None else strict_negative_leakage

    ids = [row.sample_id for row in plan.records]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")

    enrolled_phrases: set[str] = set()
    if lexicon is not None:
        enrolled_phrases.update(entry.phrase for entry in lexicon if entry.phrase)
    else:
        for row in plan.records:
            if row.subset == "positive":
                enrolled_phrases.update(row.expected_targets)

    sorted_enrolled = sorted(enrolled_phrases)
    subset_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {}
    target_count_distribution: dict[int, int] = {}
    warnings: list[str] = []
    background_leakages: list[dict[str, str]] = []
    hard_negative_leakages: list[dict[str, str]] = []

    for row in plan.records:
        subset_counts[row.subset] = subset_counts.get(row.subset, 0) + 1
        position_counts[row.position] = position_counts.get(row.position, 0) + 1
        n_targets = len(row.expected_targets)
        target_count_distribution[n_targets] = target_count_distribution.get(n_targets, 0) + 1

        carrier_phrases = row.carrier_phrases or list(row.carrier_spans.keys())
        spans, found = _find_spans(row.text, carrier_phrases)
        if found != carrier_phrases or spans != row.carrier_spans:
            raise ValueError(f"invalid carrier spans for {row.sample_id}")

        if row.subset == "positive":
            if not row.expected_targets:
                raise ValueError(f"positive sample has empty expected_targets: {row.sample_id}")
            for target in row.expected_targets:
                if target not in row.carrier_spans:
                    raise ValueError(f"expected target {target!r} missing from carrier_spans in {row.sample_id}")
        elif row.subset == "hard_negative":
            if row.expected_targets:
                raise ValueError(f"hard_negative must have empty expected_targets: {row.sample_id}")
            if row.target_phrase and row.target_phrase in row.text:
                raise ValueError(f"hard negative contains its target: {row.sample_id}")
            for phrase in sorted_enrolled:
                if phrase in row.text:
                    hard_negative_leakages.append({"sample_id": row.sample_id, "leaked_phrase": phrase})
                    if enforce_neg_leakage:
                        raise ValueError(
                            f"hard_negative sample {row.sample_id} accidentally contains enrolled phrase {phrase!r}"
                        )
                    warnings.append(
                        f"hard_negative sample {row.sample_id} contains enrolled phrase {phrase!r}"
                    )
        elif row.subset == "background":
            if row.expected_targets:
                raise ValueError(f"background must have empty expected_targets: {row.sample_id}")
            for phrase in sorted_enrolled:
                if phrase in row.text:
                    background_leakages.append({"sample_id": row.sample_id, "leaked_phrase": phrase})
                    if enforce_neg_leakage:
                        raise ValueError(
                            f"background sample {row.sample_id} accidentally contains enrolled phrase {phrase!r}"
                        )
                    warnings.append(
                        f"background sample {row.sample_id} contains enrolled phrase {phrase!r}"
                    )

    substring_conflicts = find_substring_conflicts(sorted_enrolled)
    for conflict in substring_conflicts:
        warnings.append(
            f"lexicon substring conflict: {conflict['substring']!r} is contained in {conflict['superstring']!r}"
        )

    return {
        "valid": True,
        "records": len(plan.records),
        "subset_counts": subset_counts,
        "position_counts": position_counts,
        "target_count_distribution": {str(k): v for k, v in sorted(target_count_distribution.items())},
        "substring_conflicts": substring_conflicts,
        "hard_negative_leakages": hard_negative_leakages,
        "background_leakages": background_leakages,
        "warnings": warnings,
    }
