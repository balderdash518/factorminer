# Evidence-guided generation: 使用与实现说明

这是生成环节的增强模式，不是论文原始算法。默认关闭；开启后接入 Ralph 和 Helix 的共享执行路径，不改变数值评估或因子准入门槛。

## 启用

在你当前使用的数据与模型配置中加入：

```yaml
memory:
  evidence_guided:
    enabled: true
    reserve_candidates: 2
    max_replacement_rounds: 2
    history_limit: 6
    match_limit: 3
    max_context_chars: 8000
    similarity_threshold: 0.65
    review_enabled: true
    max_review_calls: 2
    max_review_candidates: 8
    review_max_tokens: 2048
    enable_embeddings: false
```

保留原来的数据、股票池、训练区间、target 和 LLM 设置。审阅复用生成模型，但使用独立上下文；不需要另配 API 密钥。级联模型的审阅走 frontier 路径。开启前请考虑新增审阅与替换调用的费用。

离线连通性演示：

```powershell
.venv/Scripts/python.exe -m factorminer.cli -c factorminer/configs/evidence_guided.yaml -o output/evidence-guided-demo mine --mock --iterations 3 --batch-size 8
```

内置 MockProvider 不模拟研究判断，其审阅回答可能无法解析，此时明确记录 abstained 并放行软提醒；不是低质量结论。专用离线集成测试另外使用结构化脚本模型，覆盖 keep、revise、replace、引用错误和调用失败。

## 实际执行顺序

1. 在固定运行范围中检索历史，组合相关成功、失败/冗余、尚未评估记录；按篇幅筛选后注入生成 Prompt。
2. 生成目标数量加备用数量的候选。增强模式要求 JSON 实验说明；旧的编号公式格式仍兼容，但没有提供的实验声明不会自动补造。
3. 解析、可用字段检查、保守 AST 查重。完全重复和同批重复附历史证据要求替换。
4. 对结构近邻、可选向量近邻进行有预算的 LLM 审阅。近似不是硬拒绝理由；审阅不可解析、引用不存在或调用失败时弃权。
5. 审阅要求修改时，把具体历史、差异、理由与建议实验返给生成器。新提案重新通过相同规则检查。
6. 对审阅性质的提醒，生成器可提交 retention_reason 与 falsification_condition 保留有目的的对照；不能覆盖确定重复。
7. 达到替换上限后允许不足额批次，不放宽评估门槛。只有批准的候选进入原有数值评估、冗余检查与准入路径。
8. 最终准入之后把实际结果回写档案，下一轮重新检索。LLM 意见、解析失败和未运行提案不获得虚构的 IC。

同一个字段/数据配置在运行开始时固定，Agent 不需要逐条审核。因子库另外保存快照：历史冗余结论不能在库变化后永久充当硬拒绝依据。执行故障允许重试。全新运行不会自动继承旧运行；显式恢复 checkpoint 才恢复历史。

## Agent 得到什么

生成 Agent 收到原有 DSL 规则、可用字段、因子库、记忆和研究上下文，加上以下有上限的信息：

- 档案概况、上一轮生成反馈、上一批评估反馈、家族历史；
- 精选历史的公式、记录 ID、实际结果或未测标记、相关性与具体差异；
- 替换时额外收到已接受候选与本次拒绝/审阅通知。

预生成历史包、单次审阅包和单次替换通知各自受 max_context_chars 限制；它不是整个原有 Prompt 的总上限。历史证据整条纳入或整条省略；篇幅不足时不伪称完整召回。固定状态摘要可能截短。

生成器输出字段：name、formula、hypothesis、reference_ids、declared_parent_id（可选）、change_description、expected_improvement、falsification_condition。保留审阅建议下的对照时另外提供 retention_reason。

审阅 LLM 只收到当前待审候选、它匹配的历史、结构差异及任务约束。输出 action（keep/revise/replace）、reason、evidence_ids、suggested_test。引用 ID 只允许来自该候选实际提供的历史。

## 相似性与查重的区别

- 硬查重使用执行器解析后的有序 AST、算子参数和完整浮点值；忽略空格等书写差异。不使用 SymPy 等价、去掉 Neg/CsRank 或向量阈值来硬拦截。
- 结构检索比较树路径、骨架与包裹子树，返回窗口、字段、算子等变化。
- 文本检索复用 BM25，多路排序用 RRF 融合。enable_embeddings 显式开启后复用现有 FormulaEmbedder；可能需要可选依赖与模型下载。
- 相似分数是检索依据，不是有效因子概率，也不是公式等价证明。
- 本版没有新增基于实际信号面板的近邻检索，也没有训练自定义成功预测模型。实际信号相关检查仍由原有评估路径完成。
- Helix 开启本模式时绕过旧的生成阶段 SymPy/向量去重，交由统一规则检查，避免在入档前丢掉候选。未开启时保持原行为。
- 确定性 refinement 继续保留原实验意图，不额外发起 LLM 替换；仍做硬检查并留档。可选 debate 的专家会收到历史文本；其既有筛选路径不保证保留全部结构化实验声明。

显式声明的父实验与算法近邻分别保存在 declared_parent_id、nearest_prior_ids。增强模式下不再把自动相似匹配伪装成编辑父节点；确定性 refinement 的真实父公式保留。

## 产物与核对

运行目录的 attempt_archive.jsonl 保存追加事件：提案、解析排除、规则检查、相似匹配、审阅、修改要求、批准、备用未使用、实际评估、上下文选择与每轮生成反馈。

measured=false 的条目不能作为低 IC 证据；执行失败的指标记为 null。global_trial_ledger.jsonl 继续承担原有“实际接触数据”的试验账本职责，二者不混用。

trace_log.jsonl 与已有 Langfuse 集成记录：

- retrieve-attempt-history：实际选择的记录及注入文本；
- check-candidate-history：批准结果与统计；
- review-candidate-experiments：独立审阅角色；
- review-similar-attempts：每次模型调用的实际输入、输出及用量；
- propose-factors：首次和替换生成的实际 Prompt 与回复。

审阅模板加入现有受管理 Prompt 清单；挖掘只读取，不自动发布远端模板。

checkpoint 会携带当前可见的档案快照与运行标识。恢复旧 checkpoint 只重放该快照，不读取同目录在它之后产生的结果；导入新目录不覆盖已有其他运行记录。

每轮 generation_* 指标包括原始提案数、确认重复数、相似提醒数、审阅次数、修改要求数、替换轮数、批准数量与不足数量。不要仅凭过滤后的重复率证明质量改善；需要在相同评估预算下比较有效且互补因子的产出，最终测试集不能回流历史检索。

## 测试

```powershell
.venv/Scripts/python.exe -m pytest factorminer/tests/test_evidence_guided_generation.py -q
```

这些测试验证闭环、规则权限、引用、数值事实与意见分离、长度上限、调用用量、Ralph/Helix 接入和 checkpoint 隔离；不证明模型挖掘效果已经改善。
