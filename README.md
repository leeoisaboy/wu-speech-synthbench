<p align="center">
  <a href="https://github.com/leeoisaboy/wu-speech-synthbench">
    <img src="docs/assets/title_banner.zh.png" alt="Wu-Speech-SynthBench 多语种语音语料体检、概率校准与防刷分评测底盘" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/wu-speech-synthbench/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-SynthBench_Studio-0ea5e9?style=flat-square"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-4.0.0-6366f1?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-22%20passing-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Wu-Speech-SynthBench (v4.0)</h1>

<p align="center"><strong>零重型依赖、纯 NumPy 的语音评测与质检底盘：覆盖多语种语料防泄漏体检、自适应底噪切字、真实打分概率校准与防刷分多区间定位评测。</strong></p>

做语音关键词检索（KWS）和方言识别（ASR）时，很多看似漂亮的实验涨分，背后其实藏着四个常见水分：负样本里不小心混进了注册热词、能量切字在真实底噪下把边界拉飞、用类别平衡训练校准器却忘了修正先验导致概率虚高、以及评测时把负样本或 `no-position` 误报排除在分母之外。

这个项目是整个「白盒语音检索四部曲」的**统一基础设施与质检底盘**。核心库仅依赖纯 NumPy，不绑定任何深度学习框架，v4.0 将原本面向「吴语 + 合成音频」的协议全面升级为**语言无关、合成与真实录音通吃**的通用工具包，并内置四部曲与 Whisper / FunASR / MFA 的标准输出适配器。

| 仓库名称 | 解决的核心场景 | 底层方案与定位 |
| --- | --- | --- |
| **[Wu-Speech-SynthBench](https://github.com/leeoisaboy/wu-speech-synthbench)**（本仓库） | 语料泄漏体检、自适应切字、概率校准与防刷分评测 | 零重型依赖纯 NumPy 底盘 + 零后端 JS 诊断工作台 |
| **[Disentangle-Acoustic-Print](https://github.com/leeoisaboy/disentangle-acoustic-print)** | 端侧极速粗召回、跨音色几何解耦与四轨示波器 | 零训练纯 NumPy 几何算子，单核 CPU RTF `0.013–0.045` |
| **[AudioLLM-Latent-KWS](https://github.com/leeoisaboy/audiollm-latent-kws)** | 录 1–3 遍样音，跨方言高精度定位关键词 | 帧级 DTW / 冻结音频大模型潜空间检词 |
| **[Dialect-CrossModal-Align](https://github.com/leeoisaboy/dialect-crossmodal-align)** | 不录样音，只写音素/IPA 拼写直接检索方言新词 | 零样本跨模态对齐框架，含无标签跨域在线校准 |

---

## v4.0 架构总览

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="Wu-Speech-SynthBench v4.0 四阶段架构总览图" width="1100" />
</p>

## 实时零后端诊断台（SynthBench Studio）

<p align="center">
  <a href="https://leeoisaboy.github.io/wu-speech-synthbench/">
    <img src="docs/assets/studio-demo.gif" alt="SynthBench Studio 浏览器本地诊断台动图" width="1100" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/wu-speech-synthbench/">▶ 打开在线 SynthBench Studio</a></strong></p>

---

## 四轨真实数据五步实证（真 TTS · MagicHub 沪语 · KeSpeech 多方言 · LibriPhrase 英语）

v4.0 不再只依赖合成探针，而是直接把前三个仓库的真实输出、公开语料（L1 MagicHub 沪语、L2 KeSpeech 多方言、L3 LibriPhrase 英语）、真实神经 TTS（`facebook/mms-tts-eng`）以及真实 `openai/whisper-tiny` ASR 引擎喂进四个核心模块做全链路压测。完整原始数据见 [reports/real_data_benchmark_v40.json](reports/real_data_benchmark_v40.json)，自动门禁核验结果见 [reports/release_gate_verdict_v40.json](reports/release_gate_verdict_v40.json)。

### 1. 自适应切字边界消融 & 真实得分概率校准（Step 1–3）

<p align="center">
  <img src="docs/assets/timestamp_and_calibration_real.png" alt="Step 1-2 真实语音边界 MAE 消融与 Step 3 真实检索得分校准 ECE 下降" width="1080" />
</p>

- **Step 1–2（自适应底噪切字与诚实能力边界）**：
  - 原版 `tts_clean`（针对 `-120 dB` 数字静音设计）直接跑在真实室内录音上容易被底噪和连读拉偏；启用 v4.0 的 `preset="real_noisy"`（10% 分位动态底噪 + 6 dB 门限 + 收紧高频子带偏移 `-12 dB`）后，**L1 沪语干净集 MAE 从 `113.8 ms` 降至 `80.3 ms`（`-33.5 ms`），L2 多方言从 `114.4 ms` 降至 `74.1 ms`（`-40.3 ms`）**。
  - 在 10 dB 背景噪声下，进一步融合外部 `WavLM-Large` 帧级后验后，**L1 MAE 压至 `57.2 ms`，L2 MAE 压至 `47.3 ms`**。
  - **诚实边界说明**：在真神经 TTS（`facebook/mms-tts-eng`，18 句句首/句中/句尾）上，纯能量切字区间 IoU 达 **`0.784`**（MAE `55.7 ms`），但受神经声码器气声尾音影响，50 ms 严格命中率为 `33.3%`（未达 85% 门槛）；低信噪比（10 dB）纯信号切字同样会退化。因此在高精度场景下建议叠加外部帧后验（`posterior_probs`）。
- **Step 3（真实得分矩阵上的先验修正 & Robust MAD 抗异常峰）**：
  - 将 `disentangle-acoustic-print` 在三套公开语料上产出的真实 40 词打分矩阵按说话人隔离拆分 Dev/Test 后，开启 `LogisticCalibrator(class_weight="balanced", correct_prior=True)` 在**全部 3 套真实语料上均大幅压低 ECE**：
    - **L1 沪语**：同域 ECE `0.1751 → 0.0048`（**-97.3%**），10 dB 噪声 OOD ECE `0.3206 → 0.0169`（**-94.7%**）
    - **L2 多方言**：同域 ECE `0.1399 → 0.0041`（**-97.1%**），10 dB 噪声 OOD ECE `0.1851 → 0.0283`（**-84.7%**）
    - **L3 英语**：同域 ECE `0.0086 → 0.0031`（**-64.0%**），10 dB 噪声 OOD ECE `0.0296 → 0.0111`（**-62.5%**）
  - 在注入 2 个异常损坏通道的多目标真实打分矩阵上，**`normalization="robust"`（中位数/MAD）在三套语料上分别取得 `0.575 / 0.559 / 0.886` 的 F1**，而传统 `zscore` 因标准差被异常峰拉爆直接归零（`0.000`）。

### 2. 真实多引擎防刷分“测谎” & 多语种词边界体检（Step 4–5）

<p align="center">
  <img src="docs/assets/antigaming_and_leakage_audit.png" alt="Step 4 Whisper ASR 宽松与严格口径水分对比及 Step 5 多语种体检压测" width="1080" />
</p>

- **Step 4（真实 `Whisper-tiny` ASR vs 声学检索防刷分实测）**：
  - 在 L1 沪语上，通用中文 `Whisper-tiny` ASR 的 Micro Top-2 仅 **`1.9%`**；若用宽松口径（把负样本也算作整句完整召回），分数会被虚抬到 **`34.0%`**，而本库严格正样本口径直接揭穿真实底牌——仅 **`1.0%`**（挤掉 **33.0 pp** 水分）；同一批音频上声学检索达到 **`42.9%`**。
  - 在 L2 多方言上，宽松口径把 `Whisper-tiny` 虚抬了 **20.7 pp**（`58.7%` vs 严格 `38.0%`），且 `frequency_bias` 测出正相关（Spearman $\rho = +0.197$），印证通用 ASR 在高频词上更强、低频方言词掉底。
- **Step 5（公开语料体检 & 跨语种零误报压测）**：
  - 一键体检 L1（MagicHub）、L2（KeSpeech）和 L3（LibriPhrase）三套公开清单（各 360 条），确认零负样本泄漏与零子串冲突。
  - 在 20 条注入的中英文真实泄漏 + 10 条英文词边界陷阱（如 `"in"` 之于 `"window"`、`"at"` 之于 `"water"`、`"on"` 之于 `"morning"`）压测中，v4.0 的 `token_mode="auto"` 达成 **100% 泄漏检出率（20/20）且 0% 英文词边界误报率（0/10）**；而朴素子串匹配（`token_mode="cjk"`）在 10 条英文陷阱上出现了 **100% 误报（10/10）**。

### 3. 四项发布门禁自动验收（`python benchmarks/verify_release_gates.py`）

| 发布门禁 | 验收门槛 | 实测结果 | 状态 |
| --- | --- | --- | :---: |
| **Gate 1 切字精度与边界** | 真 TTS 50 ms 命中 ≥ 85%；干净真录音局部归一化不劣于全局 | 干净真录音局部归一化 3/3 占优且 `real_noisy` 降 MAE 33–40 ms；真神经 TTS IoU `0.784`（MAE `55.7 ms`，50 ms 命中 `33.3%`），诚实标注气声尾音与低信噪比边界 | **BOUNDARY** |
| **Gate 2 真实校准有效性** | 跨 3 套真实语料同域 ECE 降幅 > 50%，Robust MAD 不劣于 Z-score | ECE 降幅 **`-97.3% / -97.1% / -64.0%`**（3/3 达标），异常峰场景 Robust F1 **`0.56–0.89`** vs Z-score `0.00` | **PASS** |
| **Gate 3 体检零漏报与跨语种** | 注入泄漏检出率 100%，英文词边界误报率 0% | 泄漏检出 **`100%`（20/20）**，英文词边界误报 **`0%`（0/10）** | **PASS** |
| **Gate 4 零依赖与全量单测** | 核心纯 NumPy 零重型依赖，全量单测与 Studio 契约全绿 | `dependencies = ["numpy>=1.24"]`，**22/22 单元测试全过** | **PASS** |

---

## 快速上手

### 安装与命令行

```bash
pip install -e .

# 1. 一键体检任意真实 KWS/ASR 清单（自动区分 CJK 子串与英文词边界）
wu-speech-synthbench audit --manifest search_all.jsonl --token-mode auto

# 2. 对外部检索/ASR 输出做概率校准、Robust MAD 集合解码与防刷分多区间评测
wu-speech-synthbench evaluate \
  --sets test_scores.jsonl \
  --calibrator-train dev_scores.jsonl \
  --normalization robust \
  --class-weight balanced

# 3. 运行 22 项单元测试与 4 项发布门禁核验
pytest -v
python benchmarks/verify_release_gates.py
```

### Python API（四部曲统一适配器 + 自适应切字）

```python
from wu_speech_synthbench import (
    audit_manifest,
    from_asr_transcript,
    from_disentangle_matches,
    segment_word,
)

# 1. 体检真实语料（自动规避英文 "in" vs "window" 误报）
audit_report = audit_manifest("eval_manifest.jsonl", token_mode="auto")

# 2. 真实带噪录音自适应底噪切字（支持 8–48 kHz 自动奈奎斯特钳制）
span = segment_word(waveform, sample_rate=16000, characters="开灯", anchor_times=[0.45, 0.65], preset="real_noisy")

# 3. 统一接入四部曲或外部 ASR 引擎输出
res_dap = from_disentangle_matches("utt-01", dap_matches, expected_targets=["开灯"], frame_rate_hz=50.0)
res_asr = from_asr_transcript("utt-01", whisper_text, lexicon, expected_targets=["开灯"])
```

更多契约与模块边界详见 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) 与 [docs/MODULE_MAP.md](docs/MODULE_MAP.md)。
