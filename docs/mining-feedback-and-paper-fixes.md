# 挖掘反馈链路完善与论文一致性修复

日期：2026-10-23
对照：论文 arXiv:2602.14670v2 §3.1–3.4、算法 1、§4.1、附录 A；审计见 [paper-implementation-audit.md](paper-implementation-audit.md)、[mining-process-reproduction-gap.md](mining-process-reproduction-gap.md)；RD-Agent 参考见 [rdagent-factor-loop-reference.md](rdagent-factor-loop-reference.md)。

本次改动分两部分：(一) 让每轮评估的细节真正传回生成器（反馈内容、细节、内部信息传递）；(二) 修复挖掘流程中与论文不符的实现。旧行为都保留为可配置选项。

## 一、反馈链路：τ → 记忆 → prompt

### 改动前的问题（均已在代码中复现）

| 问题 | 位置 | 后果 |
|---|---|---|
| 轨迹 τ 优先取自 lifecycle 日志，其中 `max_correlation` 恒为 0、`correlated_with` 恒为空 | `core/ralph_loop.py::_build_trajectory`、`architecture/lifecycle.py` | 论文 §3.4 要求的冗余诊断没有进入 F |
| 相关性拒绝的原因文本是 `"Max spearman dependence ..."`，F 只认 `"correlation"` 关键字 | `memory/formation.py` | **P_fail 从不由挖掘过程学到**，只剩预置条目 |
| 验证管线从不设置 `correlated_with` | `application/validation_pipeline.py` | 无法告诉 LLM "和哪个因子撞了" |
| 形成算子以现有计数为起点，演化算子再加一次 | `formation.py` + `evolution.py` | 计数每轮 `2n + 新增` 指数膨胀，置信度/裁剪失真 |
| prompt 读 `size`/`target_size`，库摘要里是 `library_size`，且没有 target | `agent/prompt_builder.py` | 每轮都显示 `Library size: 0 / 110` |
| 推荐/禁止方向以 Python dict 原样打印，再以另一标题重复一遍；`strategic_insights` 键名对不上 | 同上 | 记忆重复、噪声大 |
| 库视角只有最近 5 个因子 | `core/factor_library.py::get_state_summary` | 论文强调的 global library perspective 缺失 |
| 关键词分类器 `"Or"` 能匹配 `Corr`，`Delta($close)` 会被标成 "VWAP Deviation" | `formation.py` | 记忆标签错误 |
| 修复 prompt 不含解析错误信息 | `agent/factor_generator.py` | LLM 只能盲改 |

### 现在的流程

1. **结构化结局**：`EvaluationResult.outcome` ∈ `parse_error / compute_error / invalid_signal / low_ic / low_icir / redundant / batch_duplicate / research_rejected / admitted / replaced`；另记 `correlated_with`（最近的库因子 `#id name`）、`conflict_count`（超过 θ 的库因子个数）、`replacement_note`（为什么能/不能替换）、`duplicate_of`（批内保留的那个候选）。
2. **完整轨迹 τ**：直接由评估结果构建，含 IC/ICIR、max ρ、冲突因子、结局、替换说明。
3. **形成算子 F**：P_fail 只收冗余类失败（论文 §3.3）；实现失败（没跑起来）不进入经验，也不参与算子/特征统计；饱和度按冗余结局计算；只输出本批增量，由 E 合并。
4. **批次反馈记录**（新增 `memory/feedback.py`，确定性、数值全部来自评估器）：各结局计数、准入率趋势、入库因子、"强但冗余"的候选及其冲突对象、最接近 IC 门槛的弱候选、未能运行的公式及错误原因、最拥挤的库因子，以及基于证据的诊断句（如"11/20 与库重复，主要撞 #1、#6"、"某候选只与 #1 冲突，更强的变体可替换它"、"连续三批零入库，需转向"）。存入 `MiningState.last_batch_feedback` / `feedback_history`，随记忆持久化。
5. **检索算子 R**：把反馈渲染进 `prompt_text`；No-Memory 消融下 F/R 关闭，反馈也随之关闭，消融仍是单变量。
6. **prompt**：库规模/目标、准入规则、按类别的组成、全部库因子（最多 40 条，含 |IC|）、记忆只渲染一次（标题改为 `## EXPERIENCE MEMORY`），dict 条目在无 `prompt_text` 时格式化为自然语言。
7. **可选 LLM 复盘**（`memory.llm_reflection: true`，默认关）：借鉴 RD-Agent 的 summarizer，LLM 读上述确定性反馈，输出 observations / hypotheses / avoid 的 JSON，附在下一轮反馈中。它不改变任何准入或记忆统计，解析失败直接丢弃。

## 二、与论文不一致处的修复

| # | 论文 | 改动前 | 现在 | 恢复旧行为 |
|---|---|---|---|---|
| 1 | 式 (3)(4)(10)：`ρ = mean_t Corr_t`，约束 `|ρ| < θ` | `mean_t |Corr_t|`（+1/−1 交替的两信号被判为 1.0） | `spearman` = `|mean_t Corr_t|`；索引缓存同时算两种聚合 | `evaluation.redundancy_metric: mean_abs_spearman` |
| 2 | 式 (10) 无 ICIR 条件 | `icir_threshold: 0.5` 硬拒绝 | 默认 0（关闭）；§4.3 消融配置同样为 0 | `mining.icir_threshold: 0.5` |
| 3 | 批内去重保留更强因子 | 按带符号 `ic_mean` 排序，强负向因子被丢弃 | 按 `ic_paper_mean`（|IC|）排序 | — |
| 4 | DSL 作用于原始字段，标准化只针对因子信号 | 原始 OHLCV 先 winsorize + 横截面 z-score | 默认不处理原始字段（缺失填充、停牌屏蔽照常） | `data.standardise_raw_features` / `winsorise_raw_features: true` |
| 5 | §4.1：2024 训练，2025 测试 | `mine`/`helix` CLI 与基准运行时循环拿到全面板，入库统计含 2025 收益 | 默认只用 `train_period`，并 purge `entry_delay + holding` 根 bar；基准随后在冻结数据集上按公式重算 | `mining.train_split_only: false` |
| 6 | 算法 1 从 `L ← ∅` 开始自演化 | 默认预装论文**最终结果**（表 6/10）里的模式 | `memory.seed: empty` | `memory.seed: paper_defaults` |
| 7 | — | Windows/GBK 下读取 UTF-8 YAML 失败 | 配置读写显式 UTF-8 | — |

另：system prompt 中窗口改为按 bar 描述（原为 "20-day"，数据是 10 分钟线），"必须外包 CsRank/CsZScore" 改为可选（论文 002、006 等因子没有外层 CsRank）。

## 仍未处理（需要你决定或另做）

- 内置 "论文 110 因子" 目录与附录 D 不一致、基准方法是代理目录（审计 P0）——属于基准/目录工作，不在挖掘循环内。
- `paper_repro.yaml` 仍继承 `gemini-2.0-flash`；论文用 Gemini 3.0 Flash，需你填入实际可用的模型 ID。
- 两阶段快速筛选仍先在全资产上算完整信号（只影响速度，不影响入库集合）。
- 记忆模式命名仍用关键词族（已加锚点约束），未改为 LLM 提炼；可通过 `llm_reflection` 观察 LLM 复盘的效果后再决定。
- `PromptContextBuilder` 计算的 family / research-cycle 文本从未进入 prompt，本次未接入（非论文内容）。

## 验证

- 新增 `factorminer/tests/test_mining_feedback.py`（17 项）：式 (3) 反例、索引与参考实现一致、冗余结局与冲突因子、|IC| 去重、P_fail 由 dependence 拒绝填充、实现失败不进 P_fail、计数不再翻倍、关键词锚点、反馈渲染、prompt 键名与单次渲染、复盘 JSON 解析、默认配置、初始空记忆。
- 全量测试（排除需要 matplotlib 的 2 个模块）：1088 通过、34 失败；这 34 个与未改动的 HEAD 完全相同，均因本机 venv 缺少 xgboost / networkx / tabulate 等可选依赖。受本次语义变化影响的 6 个旧测试已按新语义更新（去重按 |IC|、F 只输出增量、式 (3) 参考实现、记忆标题、两个极小数据集测试显式关闭 `train_split_only`）。
- mock 端到端：训练面板 259/500 期（只含 2024），3 轮内 P_fail 由真实冗余拒绝生成，prompt 显示真实库规模、库因子列表和上一批反馈。
