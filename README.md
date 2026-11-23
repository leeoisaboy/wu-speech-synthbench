<p align="center">
  <a href="https://github.com/leeoisaboy/wu-speech-synthbench">
    <img src="docs/assets/title_banner.zh.png" alt="Wu-Speech-SynthBench 零资源方言语音合成规划与评测工坊" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/wu-speech-synthbench/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-SynthBench_Studio-0ea5e9?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Quartet 2" src="https://img.shields.io/badge/四部曲_2-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/"><img alt="Quartet 3" src="https://img.shields.io/badge/四部曲_3-CrossModal_Align_Lab-0071e3?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Quartet 4" src="https://img.shields.io/badge/四部曲_4-Latent--KWS_Audit_Studio-7b6bb0?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="NumPy" src="https://img.shields.io/badge/core-pure_NumPy-2b9c94?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Wu-Speech-SynthBench</h1>

<p align="center"><strong>面向低资源方言语音的语料规划、自动切字、概率校准与多区间评测工具——全仓纯 NumPy，内置零后端交互演示台。</strong></p>


## 方法流程图 · Architecture Pipeline

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="Wu-Speech-SynthBench 零资源方言合成规划与防刷分评测架构流程图" width="1000" />
</p>

## 在线交互台：SynthBench Studio

<p align="center">
  <a href="https://leeoisaboy.github.io/wu-speech-synthbench/">
    <img src="docs/assets/studio-demo.gif" alt="SynthBench Studio 四模块完整交互演示" width="1080" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/wu-speech-synthbench/">▶ 打开在线 SynthBench Studio</a></strong></p>

上面的动图截自真实运行的静态页面，不依赖任何后端服务器，所有信号处理和评测计算都在浏览器本地完成。你可以直接拖入一段自己的 `.wav` 音频或用麦克风录两秒钟现场看切字边界，也可以亲手拖动滑块和区间框，观察每一项指标是怎么算出来的：

| 模块 | 页面里可以亲手试什么 | 对应代码 |
| --- | --- | --- |
| **01 能量谷切字显微镜** | 拖动汉字粗略锚点、切换词内局部归一化与帧中心对齐、拖入 `.wav` 或麦克风录音实时切字 | `auto_timestamp/energy_valley.py` |
| **02 集合解码与概率校准** | 给干扰词注入异常尖峰对比 Robust MAD 与 Z-score、拖动 `cap / margin / min_probability`、开关先验修正看可靠性曲线归位 | `set_prediction/multi_kw_decoder.py`、`score_vector_baseline.py` |
| **03 多区间定位与词频偏置** | 在长句时间轴上拖动预测区间或切为 `no-position`，对比宽松口径与严格口径；切换高频偏置与均衡模型看长尾词召回 | `eval_metrics/localisation.py`、`ranking.py` |
| **04 语料规划与泄漏体检** | 编辑热词表生成句首/句中/句尾覆盖矩阵，点击「一键注入隐患」查看子串碰撞与负样本泄漏扫描路径 | `synth_pipeline/manifest.py` |

<details>
<summary>点击展开四个模块的完整高清截图</summary>
<p align="center">
  <img src="docs/assets/demo-dsp.png" alt="01 能量谷切字显微镜" width="1080" />
  <img src="docs/assets/demo-calib.png" alt="02 集合解码与概率校准" width="1080" />
  <img src="docs/assets/demo-eval.png" alt="03 多区间定位与词频偏置" width="1080" />
  <img src="docs/assets/demo-plan.png" alt="04 语料规划与泄漏体检" width="1080" />
</p>
</details>

---

## 这个项目解决什么问题

做吴语等低资源方言的关键词检索（KWS）或短语定位时，团队往往既没有现成的大规模标注录音，也缺少可靠的评测基准。常见做法是先用 TTS 合成一批语料，再自动切出字级起止时间，最后用解码器和评测脚本打分。

但实际跑起来，这条链路里有五个非常隐蔽、却很容易把实验结论带偏的细节：

1. **句首尾静音会淹没词内的能量谷**：合成音频两端常常是极安静的 `-120 dB` 数字静音。如果拿整句话做全局归一化，字和字之间原本清晰的能量低谷会被压缩成一条近乎笔直的线，切字刀很容易飘偏。
2. **清擦音和送气音容易被普通带通“切残”**：读“烧水”“开灯”这类带 `sh / k` 清辅音的词时，普通语音频带（`300–3400 Hz`）里能量已经跌到底，真正的声学能量集中在 `2000–4000 Hz` 高频段。只看普通带通能量会把词首词尾的清辅音直接裁掉。
3. **类别不平衡下的概率输出容易虚高**：负样本远多于正样本时，训练分类器常用 `class_weight="balanced"` 来平衡梯度，但这会把模型输出的后验概率系统性地抬高，导致阈值失效、误报激增。
4. **宽松的评测口径容易给出“假高分”**：多词长句定位中，如果把“词对了但时间完全没重叠（`IoU = 0`）”的样本跳过不计、把无法给出位置的 `no-position` 误报从分母剔除，或者把纯负样本算进整句完整召回率，指标会比真实体验好看得多。
5. **合成语料里暗藏位置死锁与词表泄漏**：如果词表长度刚好是 3 的倍数，简单按序号取模会让某些词永远只出现在句首或句尾；而硬负样本或背景句里如果不小心混进了词表里的其他已注册热词，评测集本身就会自相矛盾。

**Wu-Speech-SynthBench** 把上面整条链路做成一个**全仓纯 NumPy、零重型模型依赖**的轻量工具包：从语料清单规划、信号级自动切字，到多词集合解码、概率校准与防刷分评测，每一步都透明、可复现，并配有浏览器白盒交互台。

---

## 整体流程与核心模块

<p align="center">
  <img src="docs/assets/pipeline.zh.png" alt="Wu-Speech-SynthBench 五个核心模块总览" />
</p>

### 1. 语料规划与泄漏体检（`synth_pipeline/manifest.py`）

拿到一份方言热词表后，规划器会生成结构化的 JSONL 合成任务清单，包含正向单目标句、多目标长句、硬负样本（近音/近形混淆词）以及背景负样本：

- **三位置均匀覆盖**：自动保证词表里的每个热词均匀分布在句首（`head`）、句中（`middle`）和句尾（`tail`），即使词表数量是 3 的倍数也不会锁死在单一位置。
- **隔离的硬负样本池**：未手动指定混淆词时，默认从未注册的独立词池中抽样，绝不借用词表里的其他热词充当负样本。
- **全词表泄漏体检（`validate_plan`）**：在真正花算力合成音频之前，先逐条扫描所有硬负样本与背景句是否意外包含任何已注册热词（`hard_negative_leakages`、`background_leakages`），并自动揪出词表内部的包含关系（例如同时注册了“灯”和“开灯”导致的 `substring_conflicts`）。

### 2. 信号级能量谷切字（`auto_timestamp/energy_valley.py`）

不需要大型强制对齐模型，纯 NumPy 数字信号处理即可把粗略的汉字锚点细化为精准的起止时间戳：

- **词内局部 Min-Max 归一化**：先由 `_refine_edge` 从首尾锚点向外搜索语音活动边界，再仅在 `[left : right + 1]` 词内区间做能量与谱通量归一化，彻底隔绝句首尾数字静音的干扰。
- **Parseval 守恒的高频子带保护**：用满足一侧实 FFT Parseval 守恒的权重计算 `2000–4000 Hz` 高频能量 `high_band_db`，使其与时域 `energy_db` 处于同一物理 dB 标尺，再取 `fused_db = max(energy_db, high_band_db - 6.0)`，稳稳保住清擦音和送气音边界。
- **帧中心后验对齐**：支持融合外部帧级后验概率，并统一按 `frame * hop + half_window` 重采样到声学帧中心（25 ms 窗长对应 12.5 ms 半窗），避免后验交叉点与声学边界之间出现系统性时间偏移。

### 3. 多词集合解码与概率校准（`set_prediction/`）

面对一句话里可能包含 0 个、1 个或多个热词的场景，工具包提供了完整的池内标准化、概率校准与集合解码流程：

- **抗尖峰标准化（`Robust Median-MAD`）**：除经典 Z-score 外，提供基于中位数与 `1.4826 × MAD` 的稳健标准化。即使某个无关干扰词突然冒出极高异常分，也不会拉歪整句基线而把真目标挤出阈值。
- **岭回归逻辑校准与先验修正（`LogisticCalibrator`）**：内置带 Armijo 线搜索的纯 NumPy 逻辑回归校准器，实时计算 `ECE` 和 `Brier Score`。启用 `class_weight="balanced"` 时，推理阶段自动执行对数几率修正：
  $$\text{logit}(p_{\text{cal}}) = \text{logit}(p_{\text{bal}}) - \ln(w_1 / w_0)$$
  将失衡权重抬高的后验概率还原到真实概率尺度。
- **排序分歧保护（`decode_set`）**：当候选按校准概率排序，而某条干扰项因 z 间隔过大被过滤时，循环使用 `continue` 跳过该项而非提前 `break`，避免误杀排在后面的高概率真目标；同时支持 `cap`、`margin`、`min_probability` 和 `prob_margin` 全网格扫描。

### 4. 多区间定位与防刷分评测（`eval_metrics/`）

用清晰、不注水的口径衡量检索与定位系统的真实表现：

- **纯 NumPy 匈牙利最优匹配（`evaluate_multi_span_localisation`）**：在多词长句上对预测区间与真实区间执行最大权二分匹配。同一词条即使预测位置完全脱靶（`IoU = 0.0`）也会如实计入 `multi_span_known_word_iou`；无法定位的 `no-position` 预测同样计入端到端 `precision_at_iou` 分母（并单独报告 `positionable_precision_at_iou` 方便诊断）。
- **只看正样本的整句完整召回（`evaluate_ranking`）**：`sentence_recall_complete` 严格以正向语句数为分母，纯负样本不会凭空送分；`pr_auc` 按相同分数分组计算，结果严格不受输入顺序影响。
- **头 / 中 / 尾词频偏置分析（`frequency_bias`）**：按词表频次分层统计 `head / body / tail` 召回率，并计算对数词频与单字召回率之间的 Pearson / Spearman 相关系数，一眼识别“微平均很高、但低频方言词已经掉底”的偏置模型。

---

## 确定性基准探针一览

以下指标来自仓库内置的确定性合成样例与 Studio 导出数据包（用于验证算法契约与数值行为，不代表真实方言业务准确率）：

| 检查项 | 输出结果 | 含义说明 |
| --- | --- | --- |
| “烧水”自动切字 | `0.2025–0.4725 s`、`0.4725–0.7625 s` | 98 帧声学特征，浏览器 JS 与 Python 切点一致 |
| 类别加权（未修正先验） | ECE `0.1878`，Brier `0.1286` | 预测概率整体高于真实命中率 |
| 启用对数几率先验修正 | ECE `0.0282`，Brier `0.0805` | 可靠性曲线紧贴 `y = x` 对角线 |
| 整句完整召回率对比 | 宽松口径 `0.68` → 严格正样本口径 `0.36` | 剥离纯负样本带来的 0.50 虚高地板 |
| 多区间匈牙利定位 | known-word IoU `0.7818`，precision `0.80` | 含 1 条 `no-position` 误报，端到端分母如实扣分 |
| 高频偏置 vs 均衡模型 | Tail 召回 `31.5%` vs `80.5%` | Micro 都在 ~85% 时，词频雷达直接暴露长尾短板 |
| 默认语料规划体检 | 27 条记录，0 冲突、0 泄漏 | 三位置均衡覆盖，负样本严格隔离 |

---

## 快速开始

### 1. 安装依赖

核心库只依赖 `numpy`（Python 3.10+）：

```bash
git clone https://github.com/leeoisaboy/wu-speech-synthbench.git
cd wu-speech-synthbench
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e . pytest
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install -e . pytest
```

### 2. 运行测试与命令行演示

```bash
# 运行全部 17 项单元测试
python -m pytest -q

# 一键运行确定性全流程演示（含 30% 多目标长句）
wu-speech-synthbench demo --multi-phrase-ratio 0.3 --output report.json

# 对自有 JSONL 集合预测结果执行校准与网格扫描
wu-speech-synthbench evaluate \
  --sets eval_sets.jsonl \
  --calibrator-train cal_sets.jsonl \
  --caps 1,2,3 \
  --margins 0.5,1.5 \
  --min-probabilities none,0.3,0.5,0.7 \
  --normalization robust
```

### 3. 本地启动 SynthBench Studio

无需后端，导出静态数据后直接用任意静态文件服务器打开即可：

```bash
python scripts/export_synthbench_bundle.py
python -m http.server 8765 --directory studio
```

浏览器访问 `http://127.0.0.1:8765`，或直接打开在线版：<https://leeoisaboy.github.io/wu-speech-synthbench/>。

---

## 仓库目录结构

```text
wu_speech_synthbench/
  synth_pipeline/
    manifest.py            三位置语料规划、多目标长句、硬负/背景生成与泄漏体检
    demo.py                内置智能家居演示词表与规划夹具
  auto_timestamp/
    energy_valley.py       零相位带通、Parseval 高频子带、词内归一化与能量谷切字
    derive_positions.py    文本/读音位置回退推导
  set_prediction/
    multi_kw_decoder.py    Robust MAD / Z-score 标准化、集合解码与工作面扫描
    score_vector_baseline.py
                           岭回归逻辑校准、先验修正、ECE 与 Brier Score
  eval_metrics/
    localisation.py        单区间与多区间匈牙利定位评测与数据契约检查
    ranking.py             排序指标、正样本完整召回、顺序无关 PR-AUC 与词频偏置
studio/
  index.html               零后端四模块交互台
  static/studio.css        浅色长卷轴与动态视觉样式
  static/studio.js         浏览器端纯 JS FFT、解码、校准、匈牙利匹配与泄漏扫描
  data/                    静态导出数据（synthbench_bundle.json）
scripts/
  export_synthbench_bundle.py  调用真实 Python 模块导出 Studio 数据包
  make_assets.py               生成小狗 Logo、双语横幅与架构图
  make_studio_demo_gif.py      通过 Playwright 录制清晰完整的多模块演示图
docs/
  DATA_CONTRACT.md         字段定义与数据契约说明
  MODULE_MAP.md            模块职责与调用关系
tests/                     17 项数值、信号处理、校准、定位、规划与 Studio 契约测试
```

---

## 低资源方言语音白盒研究四部曲

这四个开源仓库连起来，构成一条从“语料标定 → 声学几何 → 跨模态训练 → 大模型检索审计”的完整研究闭环：

| 序号 | 仓库 | 定位 | 在线交互台 |
| --- | --- | --- | --- |
| **①** | **[wu-speech-synthbench](https://github.com/leeoisaboy/wu-speech-synthbench)**（本仓库） | **质检与标定工坊**：零资源语料规划、信号级自动切字、概率校准与防刷分评测 | [SynthBench Studio](https://leeoisaboy.github.io/wu-speech-synthbench/) |
| **②** | **[disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print)** | **声学示波器**：免训练说话人解耦与等弧长语速对齐 | [Acoustic Print Studio](https://leeoisaboy.github.io/disentangle-acoustic-print/) |
| **③** | **[dialect-crossmodal-align](https://github.com/leeoisaboy/dialect-crossmodal-align)** | **跨模态实验室**：组合式零样本 OOV、语速弹性注意力与路由陷阱消融 | [CrossModal Align Lab](https://leeoisaboy.github.io/dialect-crossmodal-align/) |
| **④** | **[audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws)** | **大模型审计台**：80ms/40ms 双网格防泄漏检索、Top-20 语境重排与 KV-Cache 加速 | [Latent-KWS Audit Studio](https://leeoisaboy.github.io/audiollm-latent-kws/) |

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
