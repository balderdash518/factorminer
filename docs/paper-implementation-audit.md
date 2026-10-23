# FactorMiner 论文—实现一致性审计

日期：2026-10-13  
论文：*FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery*，arXiv:2602.14670v2，2026-08-19  
审计对象：当前仓库工作树（只读审计，未修改产品代码）

## 结论

这个项目实现了论文的**总体架构思想**，但不是论文实验的等价参考实现。更准确的定位是：

> 一个以 FactorMiner 论文为基础、加入大量工程扩展的研究平台；其默认 `paper` / `paper_repro` 路径仍存在多处会改变公式、入库结果、消融结论和性能结论的实质差异。

最严重的问题是：仓库内置并标为“论文 110 因子”的目录与论文附录 D 基本不是同一组公式。逐条从 PDF 提取 110 条公式并去除空白后比较，只有 **006、046** 两条逐字一致；001 是把论文中含义含混的 `Min/Max` 改写成更合理的 `TsMin/TsMax`；其余 **107 条是实质不同的公式**。因此，以内置目录运行的 `factor_miner` 基准不能解释为论文表 1 的复现。

## 一致的部分

| 论文要素 | 仓库实现 | 判断 |
| --- | --- | --- |
| Ralph Loop：检索、生成、评估、库更新、记忆演化 | `core/ralph_loop.py` 与 `architecture/stages.py` 有对应阶段 | 架构一致 |
| 目标库 110、每批 40 | `configs/default.yaml:24-28` | 一致 |
| A 股默认 IC 阈值 0.04、相关阈值 0.5 | `configs/default.yaml:33-40` | 一致 |
| 替换门槛 0.10、相对提升 1.3 倍、仅一个冲突因子 | `configs/default.yaml:42-46`、`architecture/geometry.py:175-207` | 一致 |
| Rank IC 为逐时点横截面 Spearman | `evaluation/metrics.py:107-133` | 一致 |
| 论文汇总 IC 为 `abs(mean(IC_t))` | `evaluation/metrics.py:200-215` | 一致 |
| 下一根 10 分钟 K 线 open-to-close 目标 | `data/tensor_builder.py:99-119` | 与论文“non-overlapping next-step return ratio”相符 |
| 2024 训练、2025 测试、Top-40 在 CSI500 冻结 | `configs/default.yaml:107-123,224-231` | 一致 |
| 论文模式关闭 Helix/因果/容量/显著性等扩展 | `configs/paper_repro*.yaml` | 一致 |

## 关键不一致

### P0：内置 110 因子不是论文附录 D 的 110 因子

- 论文附录 D 的 002 是 EMA Deviation：`Neg(Div(Sub($close, EMA($close, 10)), EMA($close, 10)))`；仓库 002 是 Volume-Weighted Momentum。
- 论文 003 是 volume delta 与 VWAP deviation 的差；仓库 003 是 Residual Volatility。
- 论文表 8 也直接把 002 命名为 “EMA Deviation”，与仓库名称冲突。
- 仓库在 `core/library_io.py:282-287` 把其列表标为 “110 factors from paper / normalized”，但全量比较只得到 2 条逐字一致。
- 用同一组确定性合成 OHLCV 数据分别执行论文公式与仓库公式，只有 006、046 输出一致；因此差异不是名称或空白问题。

**系统影响：**

1. `factor_miner` 内置基准测到的是另一组因子，不是论文发布库。
2. 相关热图、Top-40、组合 IC/ICIR、端到端模型重合度均不可与论文表格直接比较。
3. 由该目录生成的成功模式、失败区域和“相关性红海”结构也会变，进而改变后续 LLM 搜索轨迹。

### P0：基准方法是代理目录，不是论文中的真实方法

- `gplearn` 实际是手写的确定性深层公式变换，代码称其为 “mimic GP search”（`benchmark/catalogs.py:155-177`）。
- `alphaforge_style` 直接取 `PAPER_FACTORS[::2]`；`alphaagent_style` 直接取 `PAPER_FACTORS[1::2]`（`benchmark/catalogs.py:179-202`）。它们不是 AlphaForge/AlphaAgent 的训练与搜索实现，并且与被比较的 FactorMiner 目录共享候选来源。
- Alpha101 Classic 只有 12 条手写公式，Adapted 只使用 5 个窗口（`benchmark/catalogs.py:23-104`），而论文说明每条 Alpha101 最多生成 10 个窗口变体后选优。
- 默认 no-memory 基线不是关闭记忆后的同一 LLM 搜索，而是 `seed + 101` 的 200 条随机公式（`benchmark/datasets.py:156-162`）。
- 文档 `docs/upstream/reproducibility.md:235-242` 已承认这些是 partial catalogs / proxies。

**系统影响：** 表 1 的横向排名、记忆增益和统计显著性没有论文含义；尤其 AlphaForge/AlphaAgent 代理直接复用 FactorMiner 候选，会造成比较污染。

### P0：原始行情字段被横截面标准化后再执行公式

`load_runtime_dataset` 固定调用默认 `PreprocessConfig()`；默认行为包括：

- 每时点 1%/99% winsorize；
- 对 open/high/low/close/volume/amount/vwap/returns 全部做横截面 z-score；
- 对缺失值做日内前向填充和横截面中位数填充。

证据见 `data/preprocessor.py:20-54,299-350` 和 `evaluation/runtime.py:241-294`。论文只明确了对**已实现因子信号**的横截面 Spearman 评估，以及图 2 中对 realized factor signals 的标准化；没有说明先把所有原始价格与成交字段分别 z-score 后再代入公式。

例如论文的 `$close / $vwap - 1`、高低价区间位置、amount/volume 等都有明确价格/成交量含义；在两个独立 z-score 后做比值不再代表这些经济量，且分母可能靠近 0。

**系统影响：** 几乎所有因子的数值语义、排序、IC、相关性和入库结果都可能改变。这比普通参数误差更严重，因为它改变了 DSL 公式本身的经济含义。

### P0：相关性聚合公式不同

论文公式 (3) 先对每个时点求横截面 Spearman，再取时间平均：

`rho = mean_t(Corr_rank_t)`，约束使用 `abs(rho)`。

代码 `domain/dependence.py:80-122` 则在每个时点先取绝对值，再平均：

`mean_t(abs(Corr_rank_t))`。

两者并不等价。构造两个时点相关性分别为 `+1`、`-1` 的信号：论文值为 `abs(mean(+1,-1)) = 0`，当前代码值为 `mean(abs(+1),abs(-1)) = 1`。

**系统影响：** 当前实现对相关性符号随时间/状态反转的因子极端保守，会把论文认为正交的候选判为完全重复，显著缩小可入库空间、增加替换/拒绝，并改变“相关性红海”的形成速度。

### P1：代码增加了论文 admission rule 没有的 ICIR 硬门槛

论文算法 1、附录公式 (10) 的入库条件是 IC 阈值和相关阈值；没有 `ICIR >= 0.5` 条件。当前代码默认 `icir_threshold: 0.5`，并在 `application/validation_pipeline.py:242-252` 硬拒绝。

**系统影响：** 低稳定度但达到论文 IC 门槛的因子会被额外删除；库规模、组成、记忆轨迹和停止时间均会改变。它可能是合理的工程增强，但不能标为论文等价实现。

### P1：批内去重用 signed IC 排序，与其余 paper-IC 口径不一致

入库用 `ic_paper_mean = abs(mean(IC_t))`，但批内冲突时使用 `quality_attr = "ic_mean"`（`application/validation_pipeline.py:376-385`、`architecture/evaluation_kernel.py:217-245`）。因此，`ic_mean=-0.10` 的强反向因子会排在 `ic_mean=+0.05` 的较弱因子之后并可能被删除。

**系统影响：** 系统会系统性偏好正方向公式；虽然因子方向本可在组合阶段翻转，但强负向信号会在入库前丢失。

### P1：多阶段快速筛选并没有节省论文声称的主要计算

论文算法先在 `M_fast` 资产上快速评估，幸存者才在 `M_full` 上完整验证。当前实现先在全资产、全时间面板上计算完整公式信号，再切 100 个资产计算 fast IC（`application/validation_pipeline.py:145-179`）；之后立刻计算全量统计。Stage 4 没有一个在批内去重后才执行的独立全量计算阶段。

**系统影响：** 筛掉的候选仍支付了最昂贵的公式计算成本，无法据此复现论文的吞吐和“1000 因子约 6 分钟”结论。

### P1：后端与并行执行不符论文

- 论文说明 NumPy/CuPy 加速与 40-worker multiprocessing pool。
- 当前验证管线使用 `ThreadPoolExecutor`（`application/validation_pipeline.py:6,348-369`）。
- 表达式计划直接调用 NumPy `_dispatch_operator`，语义版本也标为 `numpy-expression-v1`（`core/expression_plan.py:1-60`）。
- 运行时信号键固定写 `backend="numpy"`（`evaluation/runtime.py:440-454`）。
- YAML 的 `evaluation.backend` 未传入该挖掘/严格评估执行路径；GPU/C 实现和速度微基准存在，但不是默认论文管线的执行后端。

**系统影响：** 论文的 GPU/C 和 A100 性能结论不能由该路径验证；配置切换后端可能只改变清单声明而不改变实际数值执行。

### P1：论文复现配置继承了不同的 LLM

论文附录写明 Gemini 3.0 Flash。`paper_repro.yaml` 没有覆盖 LLM，所以继承 `configs/default.yaml:128-148` 的 `gemini-2.0-flash`。

**系统影响：** 候选分布、格式有效率、记忆响应和最终库都会变化。LLM 搜索本来就具有路径依赖，这一差异不能视为数值容差。

### P1：默认“paper memory”预装了论文结果知识

新 RalphLoop 默认不是空记忆，而是 `create_default_memory()`（`core/ralph_loop.py:156`）。该函数预装论文总结出的成功模式、禁止方向和战略洞察（`memory/defaults.py:14-191`），包括论文最终库形成后才可观察到的 VWAP/标准化收益等高相关区域。

**系统影响：** 若目标是从零复现论文的自演化发现过程，这属于后验知识注入；会提高早期搜索命中率、减少对失败区域的探索，并使“记忆自己学到这些模式”的解释变得循环。若目标仅是部署论文经验，则这是合理产品选择，但应明确区分。

### P1：仓库的 memory ablation 不是论文的受控消融

论文 4.3 说明 Have Memory 与 No Memory 都在放宽后的 `|IC| > 0.02`、`theta = 0.85` 下比较。当前 `run_ablation_memory_benchmark` 复用表 1 管线，而 `benchmark/runtime.py:268-272` 只给 `factor_miner_no_memory` 改成 0.02/0.85；FactorMiner 保留默认 0.04/0.5。加之默认 no-memory 是随机公式代理而非同一 LLM 关闭 F/E/R，实验不是单变量消融。

**系统影响：** 产出的 yield、rejection rate 和 replacements 不能用来支持论文 Figure 3 的记忆因果结论。

### P1：当前 Windows/中文区域设置下复现入口直接失败

`utils/config.py:792-796` 使用未指定编码的 `open(path)` 读取含 Unicode 字符的 YAML。在本机 GBK 默认编码下，`load_config()` 抛出 `UnicodeDecodeError`。针对表达式、评估、基准的 126 个测试中，116 通过，10 个 benchmark 测试均因同一配置读取错误失败。

**系统影响：** 当前环境中 `paper_repro`、CLI benchmark 和相关测试无法启动；这不是论文算法差异，但会阻断实际复现。

## 论文没有充分规定、实现可能合理但无法证明等价的地方

1. **算子边界语义。** 论文没有规定 ties、NaN、warm-up、EMA 初值、方差自由度、安全除法等。当前 `CsRank` 按资产行序打破 ties，而 Spearman 统计用平均秩；离散/逻辑因子可能因此出现人为排序。
2. **附录算子命名本身有歧义。** 论文 001 写 `Min($close,48)` / `Max($close,48)`，但表 7 把滚动算子写为 `TsMin/TsMax`；当前 parser 的 `Min/Max` 是逐元素二元算子。仓库在自己的 001 中改成 `TsMin/TsMax`，这个改写很可能是正确意图，但直接运行论文原式会得到错误语义。
3. **数据口径。** 论文没有完整公开原始数据快照、成分股 point-in-time 规则、复权规则、停牌/缺失处理；仓库默认这些合约可保持 `unspecified`。即使代码完全一致，也无法从仓库独立复现论文表格。
4. **记忆蒸馏细节。** 论文把 Formation/Evolution/Retrieval 作为概念算子并给出例子，没有完整给出排序、合并、置信度、截断等算法；仓库选择了一套确定性启发式，属于一种可能实现，而非可证实的原实现。
5. **Prompt 与采样。** 论文未给出完整 prompt、温度、随机种子、重试与解析策略；这些都会改变 LLM 候选和路径依赖结果。

## 对整个系统的综合影响

| 影响面 | 结论 |
| --- | --- |
| 论文表 1 数值复现 | 不可声称复现；数据、公式目录、基线与执行后端均不等价 |
| 110 因子库 | 当前内置库不能称为论文附录 D 的归一化版本 |
| 因子入库/替换 | 相关性聚合、额外 ICIR 门槛、signed-IC 去重会显著改变结果 |
| 记忆有效性 | 默认带后验 seed，no-memory 又是随机代理；当前消融不能证明记忆贡献 |
| 经济解释 | 原始字段 z-score 后再代入公式，很多价格/成交量公式失去论文语义 |
| 性能结论 | fast screen、线程池和固定 NumPy 路径不能支持论文 GPU/多进程吞吐结论 |
| 工程可用性 | 平台扩展丰富、证据与清单意识较强；但 Windows 配置编码错误会阻断入口 |

## 建议的修复优先级

1. 建立一个不可变的 `paper_v2_110` 目录，逐条采用附录 D 公式；对 001 的别名修复写显式迁移记录，并加入公式哈希/快照测试。
2. 将论文模式的相关性改为 `abs(mean_t(rho_t))`；若产品想保留 `mean_t(abs(rho_t))`，给它单独名称和配置，不要称为 paper Spearman。
3. 论文模式移除 ICIR admission gate，批内去重按 `ic_paper_mean` 排序。
4. 让 raw fields 默认保持原尺度；只在因子输出或 IC 计算时做论文所需的横截面排序/标准化。把预处理规则写入 paper profile 并可关闭。
5. 实现真正的两阶段计算：先只对 `M_fast` 计算，去重后再对 `M_full` 计算；把 backend 实际传到执行器，并明确 CPU/GPU 数值一致性测试。
6. `paper_repro` 显式指定 Gemini 3.0 Flash、空初始记忆和固定 prompt/sampling 配置；另设 `paper_seeded_product` 表示产品化经验注入。
7. 基准中只把真实外部运行或其冻结产物称为 GPLearn/AlphaForge/AlphaAgent；代理必须单列，不能进入论文复现表。
8. memory ablation 使用同一候选生成器、同一阈值、同一预算和同一随机种子，只切换 F/E/R。
9. 所有 YAML/JSON 文本读取显式使用 UTF-8。

## 审计验证记录

- PDF 共 12 页；重点检查第 3–4 页方法、实验设置、4.3 消融、附录 A 与附录 D（PDF 第 11–12 页）。
- 从附录 D 自动提取 110 条公式；109 条可被当前 parser 原样解析，014 的文本抽取结果无法解析。可解析不等于语义正确，例如论文 001 的 `Min/Max` 在当前 registry 中是逐元素算子。
- 逐条字符串比较：2/110 一致（006、046）。
- 确定性合成面板执行比较：2/109 可执行对输出一致（006、046）。
- 相关性反例执行：论文聚合结果 0.0，当前 `SpearmanDependenceMetric` 结果 1.0。
- 定向测试：`test_expression_tree.py`、`test_evaluation.py`、`test_benchmark.py` 共 126 项；116 通过，10 项因 GBK 读取 UTF-8 YAML 失败。

