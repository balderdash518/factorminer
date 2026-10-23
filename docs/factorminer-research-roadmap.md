# FactorMiner：从批量公式搜索到可检验的研究循环

日期：2026-10-23。范围：架构建议与代码核查，不修改挖掘实现。

## 设计目标

保留 FactorMiner 的 DSL、全局因子库、相关性约束与经验记忆。在此基础上，以可检验假设组织实验，用确定性证据产生反馈，使反馈实际改变下一轮动作。复杂度必须通过统一评估协议下的消融获得支持。

论文复现配置应冻结；增强研究配置单独版本化。二者共享执行器和统计实现，允许入库政策不同。训练日期可以随数据更新，协议和数据可见性必须明确。

## 已有能力与真实缺口

| 位置 | 已有能力 | 后续工作 |
| --- | --- | --- |
| `core/ralph_loop.py` | 检索、生成、评估、更新、蒸馏 | 增加实验计划和证据反馈的输入输出 |
| `architecture/research_actions.py` | generate/refine/challenge/stop；决策价值、随机和 bandit；SQLite 记录 | 丰富动作，学习可复验的改进，而非仅入库数量 |
| `application/research_actions.py` | 父子公式、延迟检验、动作执行与恢复 | 从固定平滑/差分扩展到假设驱动实验；比较时使用共同有效样本 |
| `architecture/research_skills.py` | 固定编辑配方、带适用条件的经验迁移、反证降权 | 拓展到更有意义的机制实验，避免单次成功升级为普遍规则 |
| `application/research_knowledge.py` | 研究来源、假设与结果关联 | 统一内部生成假设和受控实验的证据链 |
| `memory/formation.py` | 模板/关键词分类、入库率和算子频数总结 | 区分实现失败、预测失败、冗余、证据不足；关联实验反证 |
| `evaluation/research.py` | 多目标、置信度、残差 IC、几何与滚动评估工具 | 审核时间切分和统计语义，并将证据接入决策 |
| `architecture/model_stage.py` | 定期因子与模型联合诊断 | 当前报告不修改入库；需显式连接下一轮研究计划 |

普通 mine 与启用 research planner 的路径不同：后者调用 discovery_dataset，按训练期截取并 purge；前者当前传完整面板。应把训练隔离提升为所有挖掘入口的不变量，不能依赖可选 planner。

## 建议的循环

1. 检索同市场、频率、目标、数据可用性下的证据和反证。
2. 提出可检验假设，写明预期方向、作用时间尺度和失效条件。
3. 生成实验计划：父因子、控制因子、单一改动、冻结评估协议。
4. 编译 DSL，检查单位、时间依赖、形状、缺失率和常量输出。
5. 内层修复可执行性；修复引起语义变化时创建新版本。
6. 在 discovery 内的时间折上比较，输出结构化数值证据。
7. 反馈解释假设得到支持、被反驳或证据不足；给出下一动作。
8. 按政策更新 active/reserve/archive，保存全部尝试。
9. 蒸馏有适用范围和反证的经验，再开始下一轮。

最终 holdout 不参与上述任何步骤。用于持续反馈的验证集属于开发数据，应采用时间折、试验账本与独立最终审计约束重复搜索。

## 优先改进

### 1. 结构化假设与实验

新增或扩展 ResearchHypothesis、ExperimentPlan，至少保存：

- hypothesis_id、parent_hypothesis_id、parent_factor_id、机制解释；
- 预期信号方向、持有期、适用市场、预期失效条件；
- exact_formula、控制公式、变更位置与原因；
- 数据/算子/协议版本、开发折、主要比较指标和完成规则。

示例：高成交量下，VWAP 偏离的短期反转更强。控制组为原始 VWAP 偏离，实验组仅增加相对成交量条件。平滑、换窗口、条件化应分成不同实验，才能辨认哪一改动有作用。

### 2. 两层反馈

内层反馈关注公式是否实现了假设：解析、参数、单位、因果时间依赖、NaN、近零除数、常量输出、重现性。可修复项进入 DSL 修复；实现无效不得作为经济机制失败写入记忆。

外层反馈关注假设是否获得支持：在共同资产/时间样本上比较父子因子；报告改进量、按时间折的符号一致性、延迟敏感性、换手、冗余和组合增益。

数值证据由评估模块生成，LLM 只能引用 evidence_id 解释。反馈状态包括 supported/refuted/inconclusive/implementation_failed/redundant。每条 next_action 都需指向一项待验证原因。

### 3. 分阶段验证与反证

- 所有候选：数据覆盖、有限值、常量、时间依赖、IC。
- 有潜力候选：共同样本父子比较、时间折、参数邻域、延迟与期限曲线。
- 最终候选：风险暴露、组合边际增益、成本压力、跨市场/时段迁移。

使用按时间块的重采样或 HAC 表达时间依赖的不确定性。记录全部尝试；普通置信区间不能覆盖反复搜索带来的选择偏差。不要把每一个诊断都变成硬门槛。

未来扰动测试：只改 t 之后的输入，不应改变 t 及之前的因子输出。最终测试数据被扰动后，discovery 的动作、因子库和记忆摘要应保持一致。

### 4. 全局因子库增量价值

保留单因子质量和 pairwise correlation，新增 cluster/family、残差 IC、在冻结基准组合上加入因子的验证增益。投影、方向、标准化和组合权重只在每个时间折的训练段拟合。

active 保存稳健且有增益的因子，reserve 保存有条件价值或高相关候选，archive 保存失败与退役记录。pairwise correlation 高不必删除研究记录；非线性条件价值需实验支持。允许 Pareto 选择质量、稳定性、换手和复杂度，不宜直接堆成任意加权总分。

### 5. 实验谱系与证据记忆

在现有账本上维护 hypothesis → experiment → formula → evidence → feedback → next_experiment 的关联。语法错误、经济失败、重复信号、场景限制分别存储。

每条经验保存：市场/频率/目标、数据版本、支持和反证记录、跨时段效果、有效样本、最近复核、状态和替代解释。多个相似公式来自同一面板不能当成独立验证。禁止区域应允许撤销或降权。

### 6. 研究动作和调度

逐步加入 simplify、change_window、condition、residualize、parameter_sensitivity、risk_neutralize。每个动作有明确控制组和输出证据；先实现确定性动作，再让 LLM 选择。

每轮维护生成、细化和反证的机会，保留探索概率。调度奖励使用开发时间折上的边际收益、稳定性与单位实验成本，不用最终测试收益。固定 110 个因子不应成为增强模式唯一目标。

### 7. 因子与模型反馈

先以等权/固定 Ridge 作为稳定参照；对同一因子库版本，用相同训练折、验证折、种子和超参数比较 baseline 与 baseline+candidate。再做删除因子的对照。

不要把特征重要性当作独立经济贡献，也不要同时改因子、模型和标签后把提升归因于某一部分。模型反馈应指导缺失信号类型与下一实验，而非让模型任意覆盖确定性入库规则。

### 8. 研究工作台

围绕每次实验展示假设、父子公式 diff、逐折证据、失败原因、参考实验、下一步和记忆变化。展示 IC/期限衰减、参数稳定性、族群冗余、换手成本和组合增益；保留完整时间戳、频率、单位与协议。

## 模块接口建议

优先扩展已有 ledger、ResearchHypothesisRecord、EvaluationResult，不再创建平行的追踪数据库。

四个核心对象：ResearchHypothesis、ExperimentPlan、EvaluationEvidence、ResearchFeedback。

小接口示意：

```python
plan = planner.propose(history, library_snapshot, research_scope)
evidence = evaluator.run(plan, discovery_view)
feedback = reviewer.review(plan, evidence)
decision = admission.decide(evidence, library_snapshot, policy)
memory.observe(plan, evidence, feedback, decision)
```

所有外部依赖注入。最终测试视图由独立评估入口持有。论文政策和增强政策为两个真实实现；共用公式执行与统计模块，并分别版本化。LLM 失败不应改变已经计算出的证据。

## 分阶段交付与验收

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| 0 | 所有挖掘入口训练隔离、原始字段语义、论文筛选政策修复 | 扰动未来不改变历史信号；扰动 holdout 不改变 discovery 决策 |
| 1 | 一个受控父子实验与结构化反馈 | 共同样本比较、结果可重放、数值来自 evidence |
| 2 | 反馈驱动下一动作、谱系与反证记忆 | 下一动作指向证据；实现错误不生成经济禁区 |
| 3 | 滚动开发折、reserve、组合增益 | 候选和基准在同协议上比较，开发结果冻结后才读 holdout |
| 4 | 丰富调度、跨时段迁移、模型反馈 | 同预算多种子消融验证改进；完整保存失败与中断 |

第一个里程碑只完成“一个假设 → 父因子/单一变体 → 时间折比较 → 反馈 → 下一实验”。不要同时启用全部扩展。

## 如何证明增强有效

对照同一数据/DSL/LLM/初始记忆：A 论文基础循环；B 加假设与结构化反馈；C 加受控细化和反证；D 加组合增益与调度。主要比较应等候选评估次数，补充等模型调用/token/墙钟成本视角；不同成本维度不能假定等价。

多次独立搜索，比较冻结后因子库的前向表现、稳定性、组合增益、成本、冗余和失败率，同时记录所有搜索种子。做“只记日志，不把反馈送回规划器”的消融，以验证反馈的决策价值。最终 holdout 结果只能用于本次审计；若据此修改方案，该区间已成为开发数据，下次需要新的 holdout。

## RDAgent 依据

官方循环明确包含 hypothesis_gen、hypothesis2experiment、coder、runner 和 summarizer；反馈与 Trace 可关联历史、父实验与 SOTA：

- https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/workflow/rd_loop.py
- https://github.com/microsoft/RD-Agent/blob/main/rdagent/core/proposal.py

本文的统计检验、reserve 因子库、数据隔离和接口均为针对本项目的设计建议，不是对 RDAgent 已实现能力的断言。详细官方源码核查见同目录 `rdagent-factor-loop-reference.md`。
