# 运行记录与 Langfuse 追踪

日期：2026-10-23
实现：[factorminer/utils/tracing.py](../factorminer/utils/tracing.py)；配置段 `observability`（[configs/default.yaml](../factorminer/configs/default.yaml)）。

## 1. 每次运行会留下什么

输出目录（`--output-dir`，默认 `./output`）下：

| 文件 | 内容 | 本次之前就有 |
|---|---|---|
| `trace_log.jsonl` + `system_prompts/` | **每轮每个阶段的输入/输出**：发给 LLM 的完整 prompt、LLM 原始回复、模型与参数、token 用量、每个候选的评估结局、入库结果、记忆更新与批次反馈 | 否（新增） |
| `factor_lifecycle.jsonl` | 每个候选经过的阶段事件 | 是 |
| `global_trial_ledger.jsonl` | 每个触达数据的候选：规范化公式哈希、逐期 IC 序列（防篡改链） | 是 |
| `mining_batches.jsonl`、`session_log.json`、`session.json` | 每轮汇总统计、每个候选的 IC/ICIR/相关性/拒绝原因 | 是 |
| `run_manifest.json` | 配置、协议、数据契约、记忆策略 | 是 |
| `checkpoint/` | 因子库、信号、记忆（**每次覆盖**）、循环状态 | 是 |
| `factor_library.json` | 最终因子库 | 是 |

**本次之前缺的**：prompt 和 LLM 回复从未落盘；token 用量没有记录（预算里一直是 0）；记忆每次 checkpoint 被覆盖，看不到它逐轮怎么变化。所以过去只能看到"结果"，看不到"Agent 当时看到了什么、说了什么"。现在两者都有：本地 `trace_log.jsonl` 始终开启，Langfuse 可选。

体积参考：mock 运行每轮约 15 KB（batch 12）；batch 40、真实 LLM 时估计约 40–60 KB/轮（尚未实测），100 轮约 5 MB。system prompt 只按内容哈希存一份。

## 2. 开启 Langfuse

```bash
uv pip install -e ".[observability]"      # langfuse 5.x
```

复制仓库根目录的 `.env.example` 为 `.env` 并填写（`.env` 已被 git 忽略；CLI 启动时自动读取，shell 中已有的同名变量优先；也可用 `FACTORMINER_ENV_FILE` 指定其他路径）。或者直接在 shell 中设置（Langfuse 项目 Settings → API Keys）：

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或 https://us.cloud.langfuse.com，或自建地址
```

`observability.langfuse: "auto"`（默认）在检测到这两个 key 时自动启用；`"on"` 在缺 key 时给出警告；`"off"` 关闭。Python SDK v5 需要 Langfuse 服务端 ≥ v4.59.0；自建的 Langfuse v3 需改用 SDK v4（把依赖改成 `langfuse>=4,<5`）。

实盘运行建议设 `observability.environment: "production"`，mock 与调试保持 `development`，避免测试数据混进看板。

## 3. Langfuse 里能看到什么

**一轮挖掘 = 一条 trace**（名称 `mining-iteration`），同一次运行的所有轮次共享一个 session（`factorminer-<会话时间>-<campaign 前 8 位>`），在 Sessions 视图中按顺序排列。

```
mining-iteration (agent)            输入：轮次、库规模；输出：入库公式、各结局计数、yield
├── retrieve-memory (retriever)      输出：注入 prompt 的记忆全文（P_succ/P_fail/洞察/上一批反馈）
├── generate-candidates (chain)      输出：解析后的候选公式、本轮 token 合计
│   ├── propose-factors (generation) system+user prompt、原始回复、模型、参数、token、thinking
│   └── repair-formulas (generation) 解析失败时的修复调用（附解析器报错）
├── evaluate-candidates (evaluator)  输出：每个候选的结局、|IC|、ICIR、max ρ、冲突因子、原因
├── update-library (span)            输出：入库/替换的因子
└── distill-memory (span)            输出：批次反馈、记忆版本、P_succ/P_fail/洞察
    └── reflect-on-batch (generation) 仅在 memory.llm_reflection: true 时
```

研究动作开启时，`refine-candidates`（tool）代替 `generate-candidates`。

**Trace 级 score**（每轮一组，可在 Dashboard 画趋势、在 Traces 中筛选）：

| score | 含义 | 看到异常时先查 |
|---|---|---|
| `yield_rate` | 入库数 / 候选数 | — |
| `redundancy_rate` | 冗余 + 批内重复占比 | 记忆是否在提示避开拥挤区；`retrieve-memory` 的输出 |
| `weak_signal_rate` | IC/ICIR 不达标占比 | 生成方向本身；prompt 中的推荐方向 |
| `implementation_failure_rate` | 解析/计算/无效信号占比 | system prompt 的算子说明、`repair-formulas` 是否有效 |
| `best_paper_ic` | 本轮最好的 \|IC\| | — |
| `library_size` | 本轮结束时库规模 | — |

**Tags**（创建时设定）：`loop:ralph|helix`、`mode:paper|research`、`memory:<策略>`、`provider:<模型>`、`mock`。用于对比不同配置的运行，例如筛选 `memory:none` 与 `memory:paper` 两组 session，对比 `redundancy_rate` 曲线。

**Trace metadata**：campaign_id、dataset_id、IC/相关性阈值、相关性口径、batch size、轮次。

## 4. 用这些记录找框架的改进点

几个具体的查法：

1. **记忆有没有起作用**：打开某个 session，按轮次看 `redundancy_rate`。若一直不降，打开冗余率高的那轮：`retrieve-memory` 里是否已经写了"避开 #6 附近"，而 `propose-factors` 的回复仍然生成同类公式 → 是生成没有遵从记忆；若记忆里根本没写 → 是形成/检索的问题。
2. **prompt 是否误导**：`implementation_failure_rate` 高的轮次，看 `evaluate-candidates` 中 `outcome` 为 `parse_error`/`compute_error` 的原因，再看 `repair-formulas` 是否修好。反复出现同一算子错误 → 改 system prompt 的算子说明。
3. **成本**：真实模型下 generation 带有 token 用量，Langfuse 按模型价格表计算成本；对比 `repair-formulas` 与 `propose-factors` 的成本占比，判断修复调用是否值得。
4. **配置对比**：同数据、同预算跑两组配置（不同 tag），在 Dashboard 中按 tag 分组比较 score。
5. **离线回看**（未开 Langfuse 时）：

```python
import json, collections
rows = [json.loads(l) for l in open("output/trace_log.jsonl", encoding="utf-8")]
# 每轮各结局计数
for r in rows:
    if r["name"] == "evaluate-candidates":
        print(r["iteration"], r["output"]["outcomes"])
# 第 5 轮 LLM 实际看到的 user prompt
print(next(r for r in rows if r["name"] == "propose-factors" and r["iteration"] == 5)["input"][1]["content"])
```

## 5. Prompt Management

主流程使用三个独立的 chat prompt，并通过 generation 的 `prompt` 字段关联到实际使用的版本：

| Prompt | 动态变量 | 记录节点 |
|---|---|---|
| `factorminer/propose-factors` | `batch_context`：本轮任务、记忆、因子库与输出格式 | `propose-factors` |
| `factorminer/repair-formulas` | `broken_formulas`：带解析错误的公式列表 | `repair-formulas` |
| `factorminer/reflect-on-batch` | `evidence`、`library_formulas`：评估证据与当前库 | `reflect-on-batch` |

模板保留现有指令。代码先生成动态上下文，再从 Langfuse 读取 `production` 标签的模板并编译；实际编译后的内容同时用于模型请求和 Input 记录。本地 JSONL 的 `prompt` 字段记录名称和版本。

首次发布或将代码中的模板更新同步到云端：

```bash
# 预览：只写入 output/langfuse-prompts/，不访问云端
uv run --no-sync python scripts/sync_langfuse_prompts.py
# 发布：使用 .env 中的 Langfuse 凭据，相同模板不会重复创建版本
uv run --no-sync python scripts/sync_langfuse_prompts.py --apply
```

在 Langfuse 中修改模板后，将目标版本标为 `production`。SDK 使用默认的 prompt 缓存策略，更新不保证即时生效。发布脚本以本地模板为准，重新执行 `--apply` 会将本地内容发布为 production，因此不要用它覆盖尚未同步回代码的云端编辑。

`observability.prompt_management: false` 可关闭托管模板读取，保留追踪。没有凭据、云端不可用、模板变量或消息结构不兼容时继续使用本地提示词，且不关联一个未实际使用的版本。自定义 specialist/system prompt 和变化后的特征注册表不会被默认模板覆盖。离线失败后，同一轮运行不反复请求该模板。

pytest 默认隔离真实 `.env` 和 Langfuse 凭据；真实云端检查应在 pytest 之外显式运行，并使用独立 session 和验证标签。历史 trace 不自动补写 prompt 版本。

参考：[变量编译](https://langfuse.com/docs/prompt-management/features/variables)、[关联 generation](https://langfuse.com/docs/prompt-management/features/link-to-traces)。

## 6. 边界

- 追踪只读取运行数据，不改变挖掘结果；导出失败只记日志，不中断挖掘（有测试覆盖）。
- Helix 的辩论路径（`agent/specialists.py`、`agent/critic.py`）直接调用模型，尚未单独记录为 generation；默认配置不走该路径。
- Google 提供方的 token 用量取自 `usage_metadata`，仅在 SDK 返回时记录。
- 已验证：mock 的本地记录及云端 observation 输入输出；Prompt Management 的离线回退、版本关联和测试隔离由针对性测试覆盖。实际云端验证结果保存在忽略目录 `output/verification/`，不会随仓库发布。
