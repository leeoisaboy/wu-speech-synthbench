# Data Contract (v3)

All record types use JSON Lines with UTF-8 text.

## Lexicon

Required accepted fields are any one of `phrase`, `hotword`, or `word`.

```json
{"phrase":"开灯","domain":"smart-home","frequency_class":"head","reading":"kai den"}
```

`frequency_class` (for example `head`, `body`, or `tail`) is used by `evaluate_ranking(..., lexicon=...)` for stratified recall auditing.

## Confusion Pair

```json
{"target":"开灯","confusion":"开心","note":"same-initial near phrase"}
```

The confusion phrase must occur once in every hard-negative carrier, and **no enrolled lexicon phrase** may appear in any `hard_negative` or `background` carrier (enforced by `validate_plan`).

## Synthesis Plan (`SynthesisRecord`)

`build_benchmark_plan` emits `SynthesisRecord` rows with decoupled alignment and retrieval fields:

```json
{
  "sample_id":"positive-00000",
  "subset":"positive",
  "text":"开灯，帮我看一下今天的事项。",
  "position":"head",
  "target_phrase":"开灯",
  "expected_targets":["开灯"],
  "expected_phrases":["开灯"],
  "carrier_phrases":["开灯"],
  "carrier_spans":{"开灯":[0,2]},
  "char_spans":{"开灯":[0,2]},
  "confusion_phrase":null,
  "provider":null,
  "voice_id":null
}
```

- `carrier_spans` / `carrier_phrases` record every carrier phrase present in `text` (including `confusion_phrase` on `hard_negative` and background topic phrases on `background`) using Python half-open indexing.
- `expected_targets` (aliased as `expected_phrases`) records enrolled targets expected by a retrieval system. For `hard_negative` and `background`, `expected_targets` **must** be `[]`.
- Unknown extension fields (such as `audio_path`, `sample_rate`, `duration_seconds`) appended by external synthesizers are preserved in `SynthesisRecord.extra` on deserialization (`SynthesisRecord.from_dict`).

## Intermediate Synthesized Manifest (`SynthesizedRecord`)

Produced by `SynthesisAdapter.synthesize` to bridge `synth_pipeline` and `auto_timestamp`:

```json
{
  "sample_id":"positive-00000",
  "subset":"positive",
  "text":"开灯，帮我看一下今天的事项。",
  "audio_path":"out/positive-00000.wav",
  "sample_rate":16000,
  "duration_seconds":2.16,
  "carrier_spans":{"开灯":[0,2]},
  "expected_targets":["开灯"],
  "anchors":{"开灯":[0.17,0.31]},
  "provider":"cosyvoice2-wu",
  "voice_id":"spk-01"
}
```

## Ranking Input

```json
{
  "sample_id":"sample-001",
  "subset":"positive",
  "expected_phrases":["开灯","关窗"],
  "predicted_phrases":["开灯","开门"],
  "scores":[{"phrase":"开灯","score":2.1},{"phrase":"开门","score":0.3}]
}
```

Order in `predicted_phrases` defines Top-K, MRR, MAP, and NDCG@K. `sentence_recall_complete` is computed strictly over positive utterances (`positive_records`); negative utterances (`expected_phrases: []`) contribute to `false_alarm_rate` and `exact_set_match`. Optional `scores` enables tie-invariant `pr_auc`.

## Set Input

```json
{
  "sample_id":"sample-001",
  "expected_phrases":["开灯"],
  "scores":[
    {"phrase":"开灯","score":2.1},
    {"phrase":"开门","score":0.3}
  ]
}
```

Phrase names must be unique inside one score vector.

## Localisation Input (Single-Span & Multi-Span)

Single-span record (`evaluate_localisation`):

```json
{
  "sample_id":"sample-001",
  "expected_phrase":"开灯",
  "expected_start":0.40,
  "expected_end":0.80,
  "predicted_phrase":"开灯",
  "predicted_start":0.48,
  "predicted_end":0.86,
  "char_source":"reading"
}
```

**Strict Contract Rule**: `predicted_start` and `predicted_end` **must both be `null`** whenever `char_source == "no-position"`, and must never be half-missing.

Multi-span record (`evaluate_multi_span_localisation`):

```json
{
  "sample_id":"multi-001",
  "expected_spans":[
    {"phrase":"开灯","start":0.40,"end":0.80},
    {"phrase":"关窗","start":1.50,"end":1.95}
  ],
  "predicted_spans":[
    {"phrase":"开灯","start":0.45,"end":0.82,"char_source":"reading"},
    {"phrase":"关窗","start":1.48,"end":1.92,"char_source":"reading"}
  ]
}
```

In v3, `multi_span_known_word_iou` includes same-phrase zero-overlap pairs (`IoU == 0.0`), and end-to-end `precision_at_iou` counts `no-position` predictions in `total_predicted_span_count`.
