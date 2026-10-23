# FactorMiner 挖掘流程：论文 vs 本仓库 差异与修改清单

日期：2026-10-21
论文：arXiv:2602.14670v2（KDD'26），重点对照 §3.1–3.4、算法 1、§4.1、§4.3、附录 A/C
范围：**只看"因子是怎么被挖出来的"**（生成 → 评估 → 入库 → 记忆）；GPU/C/多进程等加速实现不在范围内。
与 [paper-implementation-audit.md](paper-implementation-audit.md) 的关系：那份审计覆盖了 110 因子目录、基准代理等；本文件只保留与挖掘过程有关的项，并补充了 5 条那份审计没有的发现（标 **新**）。

## 结论

当前 `factorminer mine` / `paper_repro` 跑出的挖掘轨迹不能视为论文流程的复现，主要原因有五个：

1. **测试期泄漏（新）**：挖掘循环拿到的是 2024+2025 全面板，入库 IC 和相关性都用到了 2025 的收益。
2. **LLM 实际看到的 prompt 有 bug（新）**：库规模永远显示 `0 / 110`，记忆以 Python dict 原样输出且重复两遍，strategic insights 的字段名对不上。
3. **记忆 F/E 是硬编码关键词分类器（新）**：类别名直接取自论文**最终结果**里的表 6 和表 10，记忆只能"重新发现"论文已经报告过的方向。
4. 原始行情字段先做横截面 z-score，再代入公式计算。
5. 相关性的聚合口径与论文公式 (3) 不同；另外多了一道论文没有的 ICIR ≥ 0.5 硬门槛。

---

## 1. 论文中挖掘流程的可执行定义

| 步骤 | 论文规定 | 出处 |
|---|---|---|
| 初始化 | `L ← ∅`，记忆为输入 M（论文没有说明初值是否为空） | 算法 1 |
| Step 1 检索 | `m = R(M, L)`：自然语言模板加典型示例（推荐方向/禁止方向），**按当前库诊断与最近拒绝原因匹配** | 式 (9)、§3.3 |
| Step 2 生成 | `C ~ π(α|m)`，每批 40 个，Gemini 3.0 Flash，不微调 | 算法 1、§4.1、附录 A |
| Stage 1 | 在 `M_fast` 资产上计算，要求 `|IC| ≥ τ_IC`（A 股 τ=0.04） | 算法 1、式 (10) |
| Stage 2 | `max_g |ρ(α,g)| < θ`（θ=0.5），ρ 定义为 **mean_t 横截面 Spearman**，然后取绝对值 | 式 (3)(4)(10) |
| Stage 2.5 | 只与一个库因子冲突时，若 `IC ≥ 0.10` 且 `IC ≥ 1.3·IC(g)`，则替换该因子 | 式 (11) |
| Stage 3 | 批内两两 ρ < θ 去重 | 算法 1 |
| Stage 4 | 只对幸存者在 `M_full` 上做完整验证，并收集轨迹 τ | 算法 1 |
| Step 4 | `L ← L ∪ C_admitted` | 算法 1 |
| Step 5 | `M ← E(M, F(τ))`。F 从轨迹中提炼 P_succ（被入库的）和 P_fail（因高相关被拒的）；E 负责合并、淘汰、重分类（例如入库后发现 ρ=0.82 的 VWAP 变体会被移入 P_fail） | 式 (7)(8)、§3.3 |
| 轨迹 τ | 每个候选记录：公式、IC/ICIR、`max|ρ|`、入库或拒绝、是否触发替换 | §3.4 "Trajectory semantics" |
| 数据 | 10 分钟 K 线，**训练集 = 2024Q1–Q4**，2025 为 held-out | §4.1 |
| 消融 | No-Memory = 关闭 F/E/R，两组都使用 `|IC|>0.02, θ=0.85` | §4.3 |

---

## 2. 偏差清单（按对挖掘过程的影响排序）

### P0-1 **新**：挖掘期使用了测试期数据

- 代码：`load_runtime_dataset` 把整个时间区间打成一个张量，train/test 只作为 split 元数据保存（[evaluation/runtime.py:287-330](../factorminer/evaluation/runtime.py)）。`mine` 命令直接把 `dataset.data_tensor` / `dataset.returns` 传给 `RalphLoop`（[cli/mining_commands.py:178-185](../factorminer/cli/mining_commands.py)），基准的真实挖掘路径也是这样（[benchmark/mining_runtime.py:359-365](../factorminer/benchmark/mining_runtime.py)）。只有在开启 research planner 时，`discovery_dataset` 才会切出 train。
- 实测：用 `paper_repro.yaml` 加 mock 数据运行时，循环收到的面板是 `(50, 500)`，而 train split 只有 261 期，其余 239 期是 2025。
- 影响：Stage 1/2/2.5/3 的 IC 和相关性都包含 2025 收益。入库集合、记忆轨迹和最终库全部被测试期污染，表 1 的样本外结论因此不成立。
- 修改：论文模式下，进入循环前把 `data_tensor` / `returns` / `target_panels` 切到 `get_split("train").indices`，并在末端 purge `entry_delay + holding` 根 bar（可复用 `research_actions.discovery_dataset` 的逻辑）。再加一个断言，确认循环内 timestamps 的最大值不超过 `train_period[1]`。
- 注：如果你喂给 `mine` 的数据文件本身只有 2024 年，这一条不触发。不过基准路径一定会触发。

### P0-2 **新**：LLM 收到的 prompt 与设计意图不符

在空库和默认记忆下，把第 1 轮实际发给 LLM 的 user prompt 渲染出来检查，发现以下问题：

| 问题 | 位置 | 后果 |
|---|---|---|
| 读取 `library_state["size"]`，但 `get_state_summary()` 返回的键是 `library_size`；`target_size` 也不存在 | [agent/prompt_builder.py:283-284](../factorminer/agent/prompt_builder.py)、[core/factor_library.py:573](../factorminer/core/factor_library.py) | 每一轮都显示 `Library size: 0 / 110`，LLM 无法感知库的增长 |
| `recommended_directions` / `forbidden_directions` 是 dict 列表，被 `f"* {d}"` 直接打印 | prompt_builder.py:307-320 | prompt 中出现 `{'name': ..., 'occurrence_count': 0, 'confidence': 1.0}` 这样的 Python 原始 dict |
| 同一份记忆又通过 `prompt_text` 以 "HELIX RETRIEVAL SUMMARY" 为标题再输出一次 | prompt_builder.py:330-335 | 记忆重复两遍（默认记忆下第 1 轮 prompt 约 9.3k 字符），标题也属于另一套系统 |
| 读取 `strategic_insights`，但 `retrieve_memory` 返回的键是 `insights` | prompt_builder.py:323 | 战略洞察只能通过上面那份重复文本出现 |
| 读取 `library_state["domain_saturation"]`，但库摘要里没有这个键 | prompt_builder.py:299 | 论文 §3.3 的 "saturation metrics" 不会以库状态的形式出现 |
| 最近拒绝原因只以计数形式出现（`recent_rejections_count`），没有具体公式和原因 | [memory/retrieval.py:694-716](../factorminer/memory/retrieval.py) | 论文明确说按 "recent rejection reasons" 检索并作为约束 |
| 库视角只包含最近 5 个入库因子 | [core/factor_library.py:567](../factorminer/core/factor_library.py) | 论文强调的 "global library perspective" 在 prompt 里只剩 5 个公式 |

修改：把 `library_state` 的键统一起来（推荐改 PromptBuilder 去读取 `library_size`，并从配置中传入 `target_size`）。记忆只渲染一次，用 `_format_for_prompt` 生成的自然语言版本，删掉 dict 列表那一段。在 prompt 中给出最近 N 条拒绝的「公式 → 原因 / max ρ / 冲突因子」。另外加一个 prompt 快照测试，至少固定第 1 轮和第 2 轮的 prompt 文本。

### P0-3 **新**：记忆 Formation/Evolution 是闭合词表的关键词规则，且词表来自论文结果

- `_PATTERN_SIGNATURES` / `_FORBIDDEN_SIGNATURES`（[memory/formation.py:29-61](../factorminer/memory/formation.py)）里的 8 个成功类和 4 个禁止类，名称与论文**表 6、表 10** 逐条对应（"Higher Moment Regimes"、"VWAP Deviation variants"、"WMA/EMA Smoothed Efficiency"……），归类方式是"公式里出现至少 2 个关键词"。
- 未命中任何类别的公式会被命名为 `Novel: Op1+Op2+Op3` / `HighCorr: Op(feat)`，模板字段就是那条原公式本身，没有经过抽象，也没有自然语言描述。
- E 中的重分类按"名称有 ≥2 个词重叠"判断（[memory/evolution.py:163-220](../factorminer/memory/evolution.py)）。
- 再加上 `RalphLoop` 默认用 `create_default_memory()` 预装论文结论（[core/ralph_loop.py:156](../factorminer/core/ralph_loop.py)，注释写的却是 "Defaults to empty memory"）。
- 影响：
  - 记忆不是从轨迹中"学"出来的，而是把论文的最终发现作为先验，再用关键词去匹配。
  - "记忆让 agent 发现了高阶矩 regime 等方向"这类结论因此变成循环论证。
  - 对于你要做的系统对比，这也意味着当前实现里**记忆的抽象能力上限就是这张词表**。
- 修改：
  - 论文模式下，初始记忆设为空的 `ExperienceMemory()`。
  - F 改为让 LLM 读取轨迹 τ（公式、IC、max ρ、冲突因子、结局），输出 JSON 格式的 `{pattern, template, canonical_examples, evidence}`。这与论文 "the agent analyzes the mining trajectory" 和"自然语言模板 + 典型示例"的描述一致。
  - 保留现有的规则版本，命名为 `rule_based` 作为对照。
  - 关键词表只允许出现在 `paper_seeded` 配置里。

### P0-4：原始字段先做横截面 z-score 再代入公式（原审计已列出，确认仍然存在）

`load_runtime_dataset` 固定使用 `PreprocessConfig()`（[evaluation/runtime.py:268](../factorminer/evaluation/runtime.py)），默认 `standardise=True`，会对 OHLCV、vwap、amount、returns 逐时点做 winsorize 和 z-score（[data/preprocessor.py:51-54](../factorminer/data/preprocessor.py)）。这样一来，`Div(Sub($close,$vwap),$vwap)`、`TsMax($close,48)` 等公式的经济含义都变了，挖掘时 LLM 写出的公式和实际被评估的量不是同一个东西。
修改：论文模式下 `standardise=False`，winsorize 也关掉；只做缺失值处理，标准化仅作用于因子输出（图 2 只说对 "realized factor signals" 做标准化）。

### P0-5：相关性聚合口径（原审计已列出，确认仍然存在）

论文式 (3) 先求 `ρ = mean_t(corr_t)`，约束用 `|ρ|`。代码在每个时点先取绝对值再平均（[domain/dependence.py:115](../factorminer/domain/dependence.py) 的 `np.abs(numerator)`）。该函数同时用于 Stage 2、2.5、3 和替换判断。
修改：新增一个 `paper_spearman` 度量 `abs(mean_t(corr_t))`，论文模式使用它。保留旧的度量，改名为 `mean_abs_spearman`。

### P1-1：多出一道 ICIR 硬门槛

`icir_threshold: 0.5`，在 [application/validation_pipeline.py:247](../factorminer/application/validation_pipeline.py) 处硬拒绝；论文式 (10) 中没有这一条。参考论文表 1：RF/Alpha101 在测试期的 ICIR 只有 0.25–0.43，在 10 分钟频率下，0.5 的单因子 ICIR 门槛会大幅收窄可入库空间，并改变"相关性红海"出现的时机。
修改：论文模式下设 `icir_threshold: 0`，或把这道门槛整体关掉。

### P1-2：批内去重按 signed IC 排序

[application/validation_pipeline.py:384](../factorminer/application/validation_pipeline.py) 用 `ic_mean` 排序，强负向因子会被排在后面并被淘汰。论文模式应改为按 `ic_paper_mean` 排序。

### P1-3：LLM 与 system prompt 先验

- 模型继承的是 `gemini-2.0-flash`（default.yaml），论文用的是 Gemini 3.0 Flash。
- **新**：system prompt 用日频语言描述窗口（"20-day rolling mean"，[agent/prompt_builder.py:135](../factorminer/agent/prompt_builder.py)），但数据是 10 分钟 bar，论文因子的窗口是 6/12/24/48 根 bar。
- **新**：system prompt 中有 "Always wrap the outermost expression with CsRank/CsZScore"（:160），而论文库里的 002、006 等因子并没有外层 CsRank。
- **新**：operator 表暴露了 84 个算子（含 `CsNeutralize`、`TsLinReg*`、`KAMA`、`HMA` 等），论文表 7 写的是 "60+"，具体是哪些没有给出。
- 论文没有公开 prompt，所以这些不算"错误"，但它们是强先验，会直接改变候选分布。修改：在论文模式下用最小、中性的 system prompt（只包含字段、算子、语法、频率说明），并把完整 prompt 文本写进 run manifest。

### P1-4：Stage 顺序与 Stage 4

代码的实际顺序是：fast IC → 全资产 IC + ICIR → 相关性 → 替换 → 去重。没有一个"去重之后才执行的 Stage 4 完整验证"。忽略加速因素的话，两者的入库集合基本等价，差异只来自 ICIR 门槛（P1-1）。图 1 中 Stage 4 标的是 "Robustness & OOS Test"，**不要**把它实现成用 2025 数据做的检验，否则会再次泄漏。

### P1-5：记忆消融不是受控实验（原审计已列出）

论文模式下，消融的两组应使用同一个 LLM、同一个 prompt 骨架、同一个预算和随机种子，阈值都设为 0.02/0.85，只把 `memory.policy` 在 `paper` 和 `none` 之间切换（`NoMemoryPolicy` 已经存在，可以直接用）。不要用随机公式代理来替代 No-Memory 组。

---

## 3. 论文没说清楚、需要你自己定并写进 manifest 的点

1. **替换规则的形式**：算法 1 写的是 `Φ(α) ≥ Φ(g*) + Δ`（加性），附录式 (11) 写的是 `IC ≥ 0.10 ∧ IC ≥ 1.3·IC(g)`（乘性）。式 (11) 中冲突的判定是 `ρ > θ`（带符号），算法 1 中是 `|ρ| ≥ θ`。代码采用了式 (11) 加 `|ρ|`，建议沿用并在文档中说明。
2. **Stage 2 在哪些资产上计算**：用 M_fast 还是 M_full，论文没有交代。代码用的是 M_full。
3. **批内并行入库**：同一批候选都是对照"批次开始时的库"做判断，再统一去重。论文也没有说是否串行更新库。
4. **F/E/R 是 LLM 做还是规则做**：论文只给出概念算子和例子，从 "agent analyzes" 的措辞看更像是由 LLM 来做。
5. **初始记忆是否为空**：算法 1 只写了 `Input: M`。要做从零复现，就应当从空记忆开始。
6. **算子边界语义**：NaN、warm-up、ties、EMA 初值等，原审计已列出。

---

## 4. 修改清单（建议做一个 `paper_strict` 配置集中开关）

| # | 改什么 | 文件 |
|---|---|---|
| 1 | 循环只接收 train 段（含 purge），并加断言 | `cli/mining_commands.py`、`benchmark/mining_runtime.py`（复用 `application/research_actions.discovery_dataset`） |
| 2 | 关闭原始字段的 z-score/winsorize | `evaluation/runtime.py:268` 让 `PreprocessConfig` 可配置 |
| 3 | `paper_spearman = |mean_t ρ_t|`，并用于 Stage 2/2.5/3 | `domain/dependence.py`、`indexed_dependence_metric` |
| 4 | `icir_threshold: 0`；批内去重按 `ic_paper_mean` 排序 | `configs/paper_repro.yaml`、`application/validation_pipeline.py:384` |
| 5 | 修 prompt 的键名，记忆只渲染一次，加入拒绝原因和库视角 | `agent/prompt_builder.py`、`core/factor_library.py:get_state_summary`、`memory/retrieval.py` |
| 6 | 初始记忆为空；F 改为 LLM 提炼（规则版保留为对照） | `core/ralph_loop.py:156`、`memory/formation.py`、`architecture/memory_policy.py` |
| 7 | 显式写明 Gemini 3.0 Flash、temperature 和种子；使用中性 system prompt，并把 prompt 存档 | `configs/paper_repro.yaml`、`agent/prompt_builder.py` |
| 8 | 记忆消融只切换 `memory.policy`，阈值 0.02/0.85 | `benchmark/runtime.py` |
| 9 | YAML 读取用 UTF-8（Windows 下的阻断性问题） | `utils/config.py` |

每一步都建议补一个小测试：(1) 时间戳上界；(3) 用 +1/−1 的反例确认结果为 0；(5) prompt 快照；(6) 空记忆起步时第 1 轮 prompt 中不含论文的类别名。

---

## 5. 对照 RD-Agent：可作为创新点切入的系统设计差异

> 以下 RD-Agent 的描述基于其公开的论文与仓库（R&D-Agent / R&D-Agent-Quant），请以你实际运行的 RD-Agent 版本代码为准再核实。其中"反馈内容"一行已对照 main 分支的 `rdagent/scenarios/qlib/developer/feedback.py` 和 `rdagent/scenarios/qlib/prompts.yaml` 核实。

| 维度 | FactorMiner（论文） | RD-Agent（factor 场景） | 可能的缺陷 / 切入点 |
|---|---|---|---|
| 因子表示 | 受限 DSL 表达式树（60+ 算子） | LLM 编写的 Python/pandas 代码，由 Co-STEER 迭代修复 | FM 的可执行性强但表达受限；RD 的表达自由，代价是实现错误和隐式前视风险 |
| 单次迭代的单位 | 一批 40 个公式，**没有显式假设** | 一个假设 → 少量因子，保留"假设–实验–反馈"轨迹 | FM 的记忆存的是"公式族"，而不是"为什么有效"的机制假设，很难跨市场迁移 |
| 评估目标 | 单因子 IC 加库级相关性准入（不经过模型） | 把新因子并入基线因子集，用模型训练和回测的组合指标判断 | FM 只看边际正交，不看对组合的增量；RD 看组合增量，但成本高、噪声大 |
| 冗余控制 | 全局库约束 `max|ρ|<θ` 加替换机制 | 主要和已有 SOTA 因子做高相关（近似重复）过滤 | 两者都只用线性/秩相关，没有条件冗余或残差信息增益 |
| 记忆 | P_succ / P_fail / insights 三类结构化记忆（本仓库用规则实现） | 假设–反馈轨迹 + 代码实现知识库 | FM 记忆没有置信度和失效机制（市场漂移）；RD 的知识偏向"实现"而不是"搜索方向" |
| 反馈内容 | 每个候选的轨迹 τ：公式、IC/ICIR、max\|ρ\|、冲突因子、入库/拒绝/替换。经 F/E 压缩成方向级记忆后才交给生成器 | 每轮：假设、1–5 个因子的描述，以及合并回测的 3 个指标（IC、年化超额收益、最大回撤，当前 vs SOTA）。由 LLM 写出观察、评价、新假设和是否替换 SOTA，再以完整历史轨迹喂给下一轮 | 原始量化信号 FM 更细（逐因子，含冗余信息）；RD 的数值很少且是组合级的，但带有与假设挂钩的解释。FM 的记忆记录"哪里有效/哪里拥挤"，不记录"为什么"；低 IC 的失败也不进入 P_fail |
| 数据 / 频率 | 10 分钟 bar 日内 | 默认是 Qlib 日频 | 对比时需要把频率和数据对齐，否则结论不可比 |

本仓库复现中暴露出来、可以直接作为论证材料的设计问题：

1. **记忆的可证伪性**：论文的 F/E 没有算法细节。一旦用规则实现，词表就成了先验（P0-3）。可以把"记忆是否真正从零学到了方向"做成可检验的实验，比如空记忆对比种子记忆、词表外类别的出现率。
2. **单因子准入与组合价值脱节**：表 1 中 FactorMiner 的 Lasso/XGB 增益很小甚至为负，说明准入规则可能已经把部分组合信息过滤掉了。可以考虑基于残差 IC 或信息增益的准入。
3. **反馈在压缩中丢掉了"为什么"**：FM 的轨迹本身很细，但经过 F/E 压缩后，生成器只看到方向级的推荐/禁止列表。按论文定义，P_fail 只收"因高相关被拒"的候选，低 IC 的失败不会形成负向记忆。本仓库的情况更差：拒绝详情只以计数进入 prompt（见 P0-2）。RD-Agent 的原始数值更少（3 个组合级指标），但每轮都有绑定到假设的 LLM 反思。一个可能的方向是把两者结合：保留逐因子统计，同时让记忆条目带上机制假设和证据。
4. **静态阈值与市场异质性**：论文自己用表 2 论证不同市场需要不同阈值，但流程中阈值是固定常数。
