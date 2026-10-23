# RD-Agent 因子研究闭环参考

核验日期：2026-10-23。范围为 Microsoft 官方 RD-Agent `main` 源码；分支会变化，正式对照实验应另行锁定 commit。本文只记录已核验能力和由此提出的设计建议，不表示本仓库已经实现这些能力。

## 已核验事实

### 1. 因子实现是否正确，与研究假设是否有效，采用不同反馈

`FactorEvaluatorForCoder` 先执行实现，再检查输出值，必要时进行代码评议并作最终实现判定。反馈包含执行、值、代码、最终判定以及是否有参考实现等信息。实现成功并不意味着因子有预测价值。来源：[factor_coder/evaluators.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/coder/factor_coder/evaluators.py)。

量化研究总结器则消费实验结果与之前最佳实验，输出 `observations`、`hypothesis_evaluation`、`new_hypothesis`、`reason`、`decision`。当前代码比较 IC、含成本超额年化收益、含成本最大回撤。来源：[qlib/developer/feedback.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/developer/feedback.py)。

### 2. 假设与具体因子任务分开

`QlibFactorHypothesisGen` 使用历史假设与反馈产生新假设；`QlibFactorHypothesis2Experiment` 将假设转换为任务，保存因子名称、描述、公式与变量，并引用此前成功实验。它检查已有任务名以避免同名重复；这不等同于语义去重。来源：[factor_proposal.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/proposal/factor_proposal.py)。

### 3. 研究评判关注加入已有库后的收益

官方因子反馈提示词描述的流程是：此前成功因子保存在最佳库中，新因子与该库合并后回测，再与当前最佳结果比较。反馈同时要求解释假设是否被支持、下一步如何改进；没有成功实现的因子不能当成已验证假设。提示词会倾向接受年化收益的微小提升，因此不能将其直接视为防多重检验的统计方法。来源：[qlib/prompts.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/prompts.yaml)。

### 4. 保存研究谱系与实现经验

`Trace` 保存实验—反馈记录、父节点关系、loop 映射，并能查找祖先和最佳实验；不是仅保存一段总结文本。来源：[core/proposal.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/core/proposal.py)。

CoSTEER 的知识记录绑定任务、实现副本和反馈；检索结构包含已成功任务、达到失败次数限制的任务、先前失败轨迹和类似任务经验。来源：[CoSTEER/knowledge_management.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/coder/CoSTEER/knowledge_management.py)。

### 5. 因子与模型可以交替优化，但属于额外搜索维度

`QuantRDLoop` 明确区分因子与模型的假设转换、编码、运行和总结路径。来源：[qlib_rd_loop/quant.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/app/qlib_rd_loop/quant.py)。动作生成支持 bandit、LLM、random 三种方式；bandit 记录上轮指标与动作再决定下一步。来源：[quant_proposal.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/proposal/quant_proposal.py)。

## 对 FactorMiner 的建议（设计提议，不是 RD-Agent 事实）

优先保留 DSL、算子执行和批量筛选能力，在其外建立研究实验层：

1. `HypothesisSpec`：假设、经济机制、预期方向、预期持有期、对照因子、可证伪条件、可用字段、父实验。将“再试一个公式”变成可检查的研究问题。
2. `ImplementationFeedback`：语法/类型、时间依赖、执行异常、形状、覆盖率、NaN/Inf、常数率、数值尺度、公式含义一致性。此层失败应修实现，不把经济方向写成失败经验。
3. `ResearchFeedback`：训练内各时间折 IC、置信区间、符号稳定性、分层单调性、换手、成本、暴露、参数敏感性、相对基线增量及假设支持程度。确定性数值由代码产生，LLM 只解释和提下一步。
4. `IncrementalExperiment`：固定现有库和模型，比较 `baseline`、`baseline + candidate`、`baseline + ablation`；防止高 IC 但无增量价值的因子占满库。
5. `EvidenceMemory`：每条结论绑定数据/协议/公式版本、实验 ID、支持与反证、适用区间和状态。执行错误、实现修复、研究结论分开记忆；禁止一次失败永久否定整个机制。
6. `ExperimentTrace`：保存候选、修复与派生关系、指标、库版本、记忆变更、LLM 原回复，支持重跑与回滚。
7. 预算分配先采用规则调度：探索新机制、修复表达式、改进已有方向、验证稳健性；有足够历史后再考虑 bandit。模型共同优化放在后期，避免同时变化因子和模型导致无法归因。

这些增强均须在封闭最终测试集之外完成。以上官方源码不能证明 RD-Agent 已经自动解决数据泄漏、反复查看验证集或多重比较问题；借鉴反馈结构不应连同未经验证的评判规则照搬。
