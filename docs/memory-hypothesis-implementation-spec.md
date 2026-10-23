# 分层经验记忆与假设驱动批次：实现规格 v2

日期：2026-10-23
前置：[mining-feedback-and-paper-fixes.md](mining-feedback-and-paper-fixes.md)（结构化结局、完整轨迹 τ、批次反馈已落地）。
依据：论文 §3.3–3.4、表 6、表 10；RD-Agent 审阅 `RD-Agent/docs/research/rd-agent-trajectory-20261020-21/审计报告.md`、`rd-agent-trajectory-20261022-23/轨迹缺陷分析与修改建议.md`、`FactorMiner-适用性分析.md`。

## 修订说明（v1 → v2）

| v1 的问题 | v2 的处理 | 位置 |
|---|---|---|
| 新功能在论文模式下"默认关闭"，关闭时对那次运行的记忆机制没有任何改善 | 区分"论文 F/E/R 的一种实现"（L0–L2，经消融后可作为 paper 默认）与"增强"（预检跳过、假设组、冷却、LLM 命名） | §0.2 |
| 经验条目没有适用范围，换市场、换标签后会被当成事实 | 每条记录绑定 Scope（数据集、目标）与库版本，规律可追溯到支持/反证记录 ID | §5.1、§6 |
| 骨架交换 `Sub`、删除小常数等规则过激 | 骨架明确为有损键，只用于聚合与检索；交换只在根层做；精确重复只看规范化哈希与数值指纹 | §4 |
| "至少一个因子入库即判假设 supported"不对；示例中 0.064 > 0.061 却写成"variant 不优于 base 为 true" | 因子结局与假设结局分开；假设判定为预先登记的配对比较，含最小效应与区块 bootstrap 区间，落在噪声内即 inconclusive | §8 |
| 规律强度只看成功数；同一面板上的相似公式被当成独立证据 | 按完整尝试记录计算比率；证据数按不同的 fine 骨架计，不按公式条数计 | §6.1 |
| 锚因子被替换后，冲突记录整体迁移到新因子（相关性不可传递） | 改为标记 `needs_recheck`，再次遇到时重新计算 | §6.3 |
| 恢复 checkpoint 后可能重复累加 | 每条记录有唯一 ID，形成过程幂等 | §5.3 |
| 检索只展示成功案例 | 每个推荐方向同时展示支持、反证、边界和未决问题 | §9 |
| （v2 初稿曾加入"证据质量"四级分级与 `EvidenceClaim` 校验） | 已删除：论文没有这一层，论文模式下记忆只来自评估器数值，分级没有作用；只保留"LLM 文本标为 untested、不形成约束"一条 | §1.3 |
| 现有代码从入库因子的构成推断"非线性优于线性" | **已修复**：改为比较有/无该特征的候选的准入率，两侧各至少 5 次有效尝试 | `memory/formation.py::_derive_insights` |

---

## 0. 定位

### 0.1 和论文的关系

论文 §3.3 定义了 M = {S, P_succ, P_fail, I} 和三个概念算子 F/E/R，但没有公开归类、合并、排序算法。因此：

- 本规格的 L0–L2 是**符合论文描述的一种具体实现**：从真实评估轨迹形成成功模式与冗余区域，随库变化演化，按库状态与近期失败检索。
- 弱信号与实现错误的单独记录、评估前预检、假设组、冷却、LLM 命名是**本项目的增强**，论文没有。
- 复杂的存储不等于复现。判断标准只有一个闭环：**评估结果 → 有证据的规律 → 状态更新 → 检索 → 下一轮生成因此改变**。验收（§12）围绕这个闭环设计。

### 0.2 默认开关

| 组件 | 性质 | 默认 |
|---|---|---|
| L0 账本、L1 骨架、L2 族统计与锚簇（`memory.policy: layered`） | 论文 F/E/R 的实现 | 初期为可选策略；经 §12.3 消融证明优于现有规则版后，设为 paper 模式默认，原实现保留为 `paper_rules` |
| 评估前跳过精确重复（`precheck_duplicates`） | 增强（论文会重评） | 关闭；开启时被跳过的候选仍以 `duplicate_trial` 结局写入 τ，使 yield 可比 |
| 假设组输出、假设判定、族冷却 | 增强 | 关闭 |
| LLM 命名 | 增强 | 关闭 |

---

## 1. 结局分类与经验类别

### 1.1 实验结局（已实现）

评估器给每个候选一个结局，写在 `EvaluationResult.outcome`（`application/mining_budget.py`）。论文 §3.4 要求轨迹 τ 记录"拒绝或准入结局（含是否替换）"，这里只是把拒绝细分开：

| 结局 | 含义 |
|---|---|
| `admitted` / `replaced` | 通过 IC 门槛与相关性约束 |
| `redundant` | IC 达标，但与具体库因子冲突（max \|ρ\| ≥ θ） |
| `batch_duplicate` | 与同批更强的候选冲突（Stage 3） |
| `low_ic` / `low_icir` | 未过质量门槛 |
| `parse_error` / `compute_error` / `invalid_signal` | 公式没有按写法运行 |
| `duplicate_trial`（新，仅在开启预检时出现） | 与历史已评估公式精确重复，被跳过 |

### 1.2 经验类别与论文的对应

| 类别 | 来源 | 论文对应 |
|---|---|---|
| 成功模式 | 准入的候选 | P_succ |
| 冗余区域 | `redundant` 的候选，按冲突的库因子聚类 | P_fail |
| 弱信号、实现失败 | 其他拒绝 | 论文不进入 P_succ/P_fail；本实现只用于批次反馈与比率统计，不形成禁止区域 |

成功与冗余可以同时成立：一个族先有多个入库，随后大量变体与它们冲突。记为"该方向已被当前库覆盖"（族状态 `saturated`），这正是论文 §3.3 E 的重分类例子。

### 1.3 论文模式下的记忆只来自评估器数值

论文模式下，F/E/R 全部是确定性规则：每条经验都由评估器算出的数值（IC、ρ、计数、比率）生成，没有 LLM 文本进入记忆。因此不需要额外的证据分级。

只有开启 LLM 相关增强（`llm_reflection`、假设组、`llm_naming`）时，才有 LLM 文本进入 prompt。对它们只保留一条规则：**LLM 文本以 `untested:` 前缀渲染，不改变任何计数、状态或禁止区域。**（`llm_reflection` 已按此实现。）

## 2. 设计约束（RD-Agent 审阅教训）

| RD-Agent 现象 | 约束 |
|---|---|
| 25/34 轮用换手解释失败，反馈里没有换手数据 | C1 判定由代码计算；LLM 文本标为 untested，不形成约束（§1.3） |
| 反馈直接写死下一轮参数，Loop 27→29 窗口来回调 | C2 反馈只给方向与证据，不给参数；冷却由程序执行（§6.2） |
| Loop 9 与 Loop 17 数值相同仍重跑；截面排名变换逃过 Pearson 去重 | C3 全历史规范化哈希 + 数值指纹（§5.2） |
| 未检验的猜测下一轮写成 "failed because" | C4 同 C1：LLM 给出的原因只作为待检验想法 |
| 一次加三个因子，无法归因 | C5 假设组内变体只改一个方面，比较在共同样本上做（§8） |
| 一次失败永久封死方向 | C6 约束有状态与到期，可凭新证据重开（§6.2） |

---

## 3. 分层总览

```
L3 文本层   名称 / 一句搜索先验（模板；可选 LLM 改写，标为 untested）
L2 规律层   成功模式（族）、冗余区域（锚簇）、弱信号、实现错误、待验证想法
            —— 每条规律都能追溯到 L0 记录 ID
L1 骨架层   formula → fine / coarse 键（有损，只用于聚合与检索）
L0 账本层   CandidateRecord（不可变、唯一 ID、带 Scope）+ 数值指纹
H  假设层   HypothesisSpec（预先登记的比较）→ HypothesisRecord（代码判定）
```

每轮数据流：

1. 生成（行格式或假设组格式，§8.1）。
2. 预检（可选）：规范化哈希、数值指纹（§5.2）。
3. 评估：现有流水线。
4. 记账：结果 → `CandidateRecord`，计算 L1 键与数值指纹。
5. 形成 F：按记录 ID 幂等更新 L2；计算假设判定。
6. 演化 E：状态机、到期、`needs_recheck`。
7. 检索 R：按 scope 过滤 → 预算内渲染（§9）。

新增文件：

| 文件 | 内容 |
|---|---|
| `memory/skeleton.py` | `SkeletonKey`、`skeletonize()` |
| `memory/ledger.py` | `Scope`、`CandidateRecord`、`CandidateLedger`、`signal_fingerprint()` |
| `memory/layered.py` | `PatternStats`、`AnchorCluster`、`LayeredMemory`、`form_layered()`、`evolve_layered()` |
| `memory/hypothesis.py` | `HypothesisSpec`、`ComparisonSpec`、`HypothesisRecord`、`judge_hypothesis()` |
| `agent/hypothesis_output.py` | 假设组 prompt 片段与解析 |
| `architecture/memory_policy.py` | `LayeredMemoryPolicy`，注册名 `layered` |

---

## 4. L1 骨架层：`memory/skeleton.py`

**骨架是有损键，只用于聚合与检索，永远不用于改写执行公式或判定等价。** 精确重复只由规范化哈希（`core/canonicalizer.py`，SymPy 代数等价）和数值指纹（§5.2）判定。

### 4.1 接口

```python
SKELETON_VERSION = "skeleton-v2"

@dataclass(frozen=True)
class SkeletonKey:
    fine: str
    coarse: str
    feature_classes: tuple[str, ...]
    core_operators: tuple[str, ...]
    depth: int
    lossy_steps: tuple[str, ...]   # 实际应用了哪些有损规则，便于审计

def skeletonize(formula: str, *, window_edges: tuple[int, int] = (6, 24)) -> SkeletonKey | None
```

### 4.2 规则（顺序执行）

1. **只剥根层**的 `Neg`、`CsRank`、`CsZScore`、`CsDemean`、`CsScale`、`Mul(正常数, x)`。理由：准入用 \|mean IC\| 与 \|ρ\|，根层的这些变换不改变横截面排序或只翻转符号。内层不剥。
2. **根层 `Sub` 交换**：仅当（第 1 步之后的）根节点是 `Sub` 时，把两个子节点排序（整体只差符号）。内层 `Sub` 保持原顺序：`IfElse(Greater(Sub(a,b),0), …)` 与交换后的版本是不同的函数。
3. 交换律算子 `Add`、`Mul`、`Max`、`Min`（逐元素）、`Corr`、`Cov` 在任意层排序子节点（这些是真正的交换律，无损）。
4. 特征归类：`$open,$high,$low,$close,$vwap → PRICE`；`$volume,$amt → VOL`；`$returns → RET`；其他字段保持原名。
5. 窗口分桶：`OperatorNode.params["window"]`（注册表中 41 个算子使用此参数名）→ `w:S` / `w:M` / `w:L`；`exponent`、`q`、`lower/upper`、`n_bins` 按常数处理。
6. 常数 → `c`。绝对值 < 1e-3 的加性项记为 `c_eps`（**不删除**，近零分母的稳定项可能影响结果；它只是不区分具体数值）。
7. `fine` = 序列化结果；`coarse` = 去掉 `Neg/Abs/Add/Sub/Mul/Div/Cs*` 后的算子集合 + 特征类集合。

### 4.3 必须满足的样例

| 公式 | fine | 说明 |
|---|---|---|
| `Neg(CsRank(Delta($close, 5)))` | `Delta(PRICE,w:S)` | 根层剥离 |
| `CsZScore(Delta($vwap, 6))` | `Delta(PRICE,w:S)` | 与上一行同组，但仍分别评估 |
| `Neg(Div(Sub($close, $vwap), $vwap))` | `Div(Sub(PRICE,PRICE),PRICE)` | 内层 Sub 保持顺序 |
| `Sub($vwap, $close)` 与 `Sub($close, $vwap)` | 相同 | 仅根层交换 |
| `IfElse(Greater(Sub($close,$open),0), $volume, 0)` 与交换 Sub 后的版本 | **不同** | 内层不交换 |
| `Corr($volume, $close, 10)` 与 `Corr($close, $volume, 10)` | 相同 | 真交换律 |
| `Div(X, Add(Y, 1e-8))` 与 `Div(X, Y)` | **不同**（前者含 `c_eps`） | 不删稳定项 |

---

## 5. L0 账本层：`memory/ledger.py`

### 5.1 记录（`memory_ledger.jsonl`，追加写）

```python
@dataclass(frozen=True)
class Scope:
    dataset_id: str        # architecture.trial_ledger.dataset_scope_identity
    target: str            # 默认目标名 + entry_delay/holding
    split: str             # "train"
```

只有 `dataset_id` 与 `target` 都相同的记录才互相比较、汇总。换数据或换目标后，旧记录不参与当前运行的去重与统计。

```python
@dataclass(frozen=True)
class CandidateRecord:
    schema: str                  # "candidate-record-v2"
    record_id: str               # stable_digest(campaign_id, iteration, position, canonical_hash)
    scope: Scope
    iteration: int
    name: str
    formula: str
    canonical_hash: str          # trial_ledger.canonical_formula_identity
    fingerprint: str | None      # §5.2；实现失败为 None
    skeleton_fine: str
    skeleton_coarse: str
    outcome: str
    paper_ic: float | None
    ic_mean: float | None        # 带符号，用于方向判断
    paper_icir: float | None
    ic_win_rate: float | None
    max_correlation: float | None
    anchor_factor_id: int | None
    conflict_count: int
    replaced_factor_id: int | None
    trial_id: str | None         # 对应 TrialLedger 记录，逐期 IC 序列从那里取
    hypothesis_id: str | None
    variant_role: str | None     # "base" | "variant" | None
    variant_change: str | None
    parent_formula: str | None
    library_version: int
```

`FactorLibrary` 增加 `version`：每次 admit / replace / remove 时 +1。

### 5.2 精确重复：规范化哈希 + 数值指纹

- **规范化哈希**：代数等价（交换、双重取负、x/x）。已有。
- **数值指纹**（新，`signal_fingerprint(signals)`）：在训练段固定抽样的诊断子面板上（固定种子选 `min(200, T)` 个时点 × 全部资产），每个时点做横截面平均秩，取符号无关形式（若第一个非零元素为负则整体取负），量化到 1e-6 后做 BLAKE2 哈希。截面排名、`mean_t(x) - x` 这类不改变排序的变换得到相同指纹——这正是 RD-Agent Loop 9/17 那类重复。
- 两者都只在**相同 `scope.dataset_id` 与 `scope.target`** 下比较。

### 5.3 `CandidateLedger`

```python
class CandidateLedger:
    def append(self, records: Iterable[CandidateRecord]) -> list[CandidateRecord]:
        """跳过已存在的 record_id，返回实际新增的记录（形成过程只消费返回值 → 幂等）。"""
    def by_hash(self, h: str, scope: Scope) -> list[CandidateRecord]: ...
    def by_fingerprint(self, fp: str, scope: Scope) -> list[CandidateRecord]: ...
    def by_fine(self, key: str, scope: Scope) -> list[CandidateRecord]: ...
    def get(self, record_id: str) -> CandidateRecord | None: ...
    def precheck(self, candidates, *, scope: Scope, library_ids: set[int]
                 ) -> tuple[list[tuple[str, str]], list[CandidateRecord]]: ...
```

`precheck` 只按规范化哈希判断（指纹要先算信号，属于评估后；评估后指纹命中的候选结局改写为 `duplicate_trial` 并附原记录 ID）：

- 历史结局为质量类或实现失败：跳过（同一 scope 下 IC 不随库变化）。
- 历史为 `redundant`：锚因子仍在 `library_ids` 中则跳过，否则允许重评。
- 历史为 `admitted`/`replaced`：跳过。

---

## 6. L2 规律层：`memory/layered.py`

### 6.1 成功模式：族统计（`PatternStats`，key = coarse）

```python
@dataclass
class PatternStats:
    coarse: str
    scope: Scope
    record_ids: list[str]                  # 全部有效尝试（证据可追溯）
    fine_outcomes: dict[str, Counter]      # fine 骨架 → 结局计数
    admitted_ids: list[str]
    redundant_ids: list[str]
    weak_ids: list[str]
    broken_count: int                      # 实现失败只计数，不进比率
    library_factor_ids: list[int]
    status: str = "candidate"
    status_since: int = 0
    open_questions: list[str] = field(default_factory=list)   # 由待验证想法挂入
```

派生量：

- 尝试数 `attempts = len(record_ids)`；**独立证据数** `n_eff = 不同 fine 骨架的个数`。同一面板上同一骨架的多个窗口变体高度相关，只算一份。
- `yield = admitted / attempts`；`recent_yield` 与 `recent_redundant_rate` 用最近 `window` 条记录计算。
- 状态转换按 `n_eff` 判证据充足，不按条数。

### 6.2 状态机（演化 E 的核心）

```
candidate ──(n_eff ≥ min_evidence 且 recent_yield ≥ promote_yield)──────▶ confirmed
candidate ──(n_eff ≥ min_evidence 且 recent_yield == 0 且弱信号占多数)──▶ low_yield
confirmed ──(recent_redundant_rate ≥ saturate_rate)─────────────────────▶ saturated   （"已被当前库覆盖"）
saturated ──(锚因子大多已不在库中)─────────────────────────────────────▶ candidate   （needs_recheck）
saturated / low_yield ──(cooldown 轮后，且新假设 reopens 点名并给出评估数据作为理由)──▶ candidate
任意 ──(idle_expiry 轮无尝试)──▶ 状态不变，检索优先级降低
```

默认值：`min_evidence=3`（按 n_eff）、`promote_yield=0.25`、`saturate_rate=0.7`、`window=8`、`cooldown=5`、`idle_expiry=15`。这些是起步值，§12.2 用回放数据校准。

- `saturated` 渲染在 FORBIDDEN 段，措辞为"已被当前库覆盖（锚：#6, #9），继续变体大概率冗余；若要替换，需 \|IC\| ≥ 1.3 × 锚因子"。这对应论文 E 的重分类例子，但依据换成真实冗余比例。
- `low_yield` 不进 P_fail，渲染在"低产出方向"，附重检条件。

### 6.3 冗余区域：锚簇（`AnchorCluster`，key = 锚因子 ID）

```python
@dataclass
class AnchorCluster:
    anchor_factor_id: int
    anchor_formula: str
    scope: Scope
    member_fine: dict[str, list[str]]     # fine 骨架 → record_ids
    rho_values: list[float]
    status: str = "active"                # active | needs_recheck | retired
    last_iteration: int = 0
```

- 形成：每条 `redundant` 记录按 `anchor_factor_id` 计入。`batch_duplicate` 只有当被保留者最终入库时，才以其库 ID 计入。
- **锚因子被替换或删除**：簇整体标为 `needs_recheck`，不迁移到新因子（A~B、B~C 不保证 A~C）。检索时以"曾与已移除的 #6 冲突，待复核"展示，不作为禁令。某个成员骨架再次被评估时，以新的 max ρ 结果更新：若与新因子冲突，加入新因子的簇；否则从禁区移除，并生成一条 counter 证据。
- `needs_recheck` 超过 `idle_expiry` 轮 → `retired`。

### 6.4 战略洞察 I

只保留比率比较（已在 `_derive_insights` 中实现批内版本）。跨批版本由 L2 族统计计算：在相同 scope 下，比较有/无某特征的有效尝试的准入率，要求两侧各 ≥ 5 个独立证据（n_eff）、差值 ≥ 0.2，陈述中写明两侧计数。

### 6.5 持久化

`ExperienceMemory` 新增 `layered: dict` 字段，`from_dict` 缺字段时为空，旧 checkpoint 可加载。`layered` 策略下，旧的 `success_patterns / forbidden_directions / insights` 列表由 L2 派生生成，KG 等旧检索路径仍可读取。

---

## 7. L3 文本层

- 默认：模板命名，如 `Skew-conditioned IfElse on RET`、`Near #6 vwap_deviation`；先验句只引用 values。
- 可选 LLM 命名（`llm_naming`）：族首次 confirmed/saturated、锚簇碰撞首次 ≥ 5 时触发，每轮 ≤ 2 次。输出 `{"name", "search_prior"}`；校验为不含数字、提到至少一个骨架中的算子或特征类，不通过则用模板文本。它不改变任何计数和状态。

---

## 8. 假设层 H（增强）

### 8.1 输出格式（`generation.output_format: hypothesis_groups`）

```json
{"groups": [
  {"hypothesis": {
     "id": "H1",
     "mechanism": "高成交量时 VWAP 偏离的短期反转更强",
     "comparison": {"metric": "abs_ic", "variant": "vwap_dev_hivol", "versus": "vwap_dev_base",
                    "direction": "greater"},
     "expected_sign": "negative",
     "reopens": null
   },
   "factors": [
     {"name": "vwap_dev_base",  "role": "base",    "formula": "Neg(Div(Sub($close,$vwap),$vwap))"},
     {"name": "vwap_dev_hivol", "role": "variant", "changes": "volume condition",
      "formula": "IfElse(Greater($volume, Mean($volume,24)), Neg(Div(Sub($close,$vwap),$vwap)), 0)"}
  ]}
]}
```

- `comparison` 是**预先登记**的机器可执行比较，代替自由文本的 `falsified_if`。`metric` ∈ {`abs_ic`, `ic_mean`, `icir`, `max_correlation`}，`direction` ∈ {`greater`, `less`}。
- 每组恰有一个 base；每个 variant 用 `changes` 说明只改的一个方面。
- 解析失败回退到旧的行格式（`hypothesis_id = None`）。

### 8.2 两种结局分开

| | 由谁决定 | 取值 |
|---|---|---|
| **因子结局** | 评估流水线（不变） | §1.1 |
| **假设结局** | `judge_hypothesis()` | `supported` / `contradicted` / `inconclusive` / `untestable` |

base 入库不能证明"条件化更有效"；variant 被判冗余也不否定"条件化提高了 IC"。两个结局独立记录、独立渲染。

### 8.3 判定算法

```python
@dataclass(frozen=True)
class ComparisonSpec:
    metric: str; variant: str; versus: str; direction: str

def judge_hypothesis(spec, records, *, trial_ledger, min_effect: float,
                     n_boot: int = 2000, block: int = 20, seed: int = 0) -> HypothesisRecord
```

1. 取 variant 与 versus 两条记录；任一为实现失败 → `untestable`（附实现错误证据）。
2. 从 `TrialLedger` 取两者的逐期 IC 序列（`compute_factor_stats` 返回的 `ic_series`），**只保留两者都有效的时点**（共同样本）；共同时点 < 60 → `inconclusive`（证据不足）。注意：在快速筛选（M_fast）阶段被拒的候选没有全面板序列（账本状态为 `screened_or_failed`）。假设组内的公式应在全资产上补算序列——实现为 `ValidationPipeline.evaluate_candidate(..., fast_screen=False)` 只对组内成员调用，或在假设模式下设 `fast_screen_assets: 0`；两种做法都要在运行清单中写明。
3. 在共同样本上计算 metric 差值 Δ（`abs_ic` 为 \|mean IC_v\| − \|mean IC_b\|；`icir` 同理；`max_correlation` 直接用记录值，不做 bootstrap，只报告直接计算的差值）。
4. 对逐期差值做循环区块 bootstrap（`block` 期一块，固定种子）得到 95% 区间。
5. 判定：
   - 区间整体在预期方向一侧，且 \|Δ\| ≥ `min_effect` → `supported`；
   - 区间整体在相反一侧，且 \|Δ\| ≥ `min_effect` → `contradicted`；
   - 其他 → `inconclusive`（包括"差值在噪声内"）。
6. `expected_sign` 单独检查：variant 的 `sign(ic_mean)` 与预期不符且 \|IC\| ≥ τ/2 时，附加一条"方向与预期相反"的说明，不改变比较结论。

`min_effect` 默认 0.005（\|IC\| 单位），可配。区间只反映固定数据上的抽样不确定性，不校正多轮自适应搜索；渲染时注明。

### 8.4 示例（修正 v1 的错误）

base \|IC\| = 0.061，variant \|IC\| = 0.064，`direction: greater`，`min_effect = 0.005`：

- Δ = +0.003 < min_effect → **inconclusive**（差值在设定的最小效应以内），不能写成"variant 不优于 base"，也不能写成 supported。
- 若两者都与 #6 冲突（ρ ≥ 0.83），另记两条因子结局 `redundant`，并附加说明"条件化未离开 #6 的冗余区"。

### 8.5 进入下一轮

```
H2 "高成交量时 VWAP 偏离反转更强" -> inconclusive (Δ|IC| +0.003, 95% CI [-0.002, +0.008], 未达最小效应 0.005)
   factor outcomes: base redundant (rho 0.86 with #6), variant redundant (rho 0.83 with #6)
```

- 不输出原因。LLM 复盘中的解释以 `untested:` 前缀渲染，并挂到对应族的 `open_questions`。
- 同一族累计 `contradicted` ≥ 2 次，或族状态为 `saturated`，才触发冷却；`inconclusive` 不触发冷却（证据不足不应关闭方向）。

---

## 9. 检索与渲染

每个推荐方向固定五行，支持与反证同时出现：

```
[P_succ] Skew-conditioned IfElse on RET   confirmed | 12 tries over 5 distinct structures, 5 admitted, recent yield 38%
  supported by : #31 IfElse(Greater(Skew($returns,24),0), ...) |IC| 0.058 ; #44 ...
  boundary     : 4 variants redundant with #31 (rho 0.71-0.88); window w:L variants all weak (0/3)
  open question: untested: whether the condition works on VOL features
  next         : complement #31 with a different conditioning variable; do not vary windows only
[P_fail] Near #6 vwap_deviation   active | 23 collisions from 3 structures, rho 0.62-0.97
  structures   : Div(Sub(PRICE,PRICE),PRICE) x14, Sub(PRICE,PRICE) x6, ...
  replacement  : needs |IC| >= 0.078 (1.3 x #6) and conflict with #6 only
```

| 段落 | 选择规则 | 默认上限 |
|---|---|---|
| RECOMMENDED | scope 兼容、status ∈ {confirmed, candidate 且 n_eff ≥ 2}，按 `recent_yield × (1 − recent_redundant_rate)` 降序；至少留 1 个 candidate 名额给证据少的族（避免只固守最成功结构） | 6 条 / 2400 字符 |
| FORBIDDEN | active 锚簇 + saturated 族，按碰撞数降序；needs_recheck 单独一行列出 | 8 条 / 1600 字符 |
| LOW-YIELD | low_yield 族，附重检条件 | 5 条 / 600 字符 |
| HYPOTHESIS RESULTS | 最近 2 轮 | 6 条 / 1500 字符 |
| FEEDBACK FROM LAST BATCH | 已实现 + "Already tried"（duplicate_trial） | 2500 字符 |

每条在 prompt 末尾附证据 ID（`[r:ab12, r:cd34]`），便于从 prompt 反查账本。

---

## 10. 配置

```yaml
memory:
  policy: "paper"             # 新增 "layered"；§12.3 消融通过后改为 paper 默认，旧实现更名 "paper_rules"
  layered:
    window_edges: [6, 24]
    min_evidence: 3           # 按独立结构数 n_eff
    promote_yield: 0.25
    saturate_rate: 0.7
    window: 8
    cooldown: 5
    idle_expiry: 15
    precheck_duplicates: false
    fingerprint_periods: 200
    llm_naming: false
    prompt_budget_chars: 8600
generation:
  output_format: "lines"      # "lines" | "hypothesis_groups"
  max_groups: 6
hypothesis:
  min_effect: 0.005
  bootstrap_block: 20
  bootstrap_samples: 2000
  min_common_periods: 60
```

---

## 11. 集成点

| 位置 | 改动 |
|---|---|
| `architecture/memory_policy.py` | `layered` → `LayeredMemoryPolicy`；`NoMemoryPolicy` 不形成 L2/H，L0 照常记账 |
| `core/factor_library.py` | `version` 计数 |
| `core/ralph_loop.py::_stage_evaluate` | 可选预检；评估后写 L0（含指纹，需要信号，故在 `signals` 释放前计算） |
| `core/ralph_loop.py::_build_trajectory` | 加 `record_id`、`hypothesis_id`、`variant_role`、`skeleton_*` |
| `agent/factor_generator.py` | 按 `output_format` 选择 prompt 片段与解析器 |
| `memory/memory_store.py` | `layered` 字段 |
| checkpoint 恢复 | 从 `memory_ledger.jsonl` 重建索引；L2 状态从 memory.json 读，不重放（避免重复累加） |

---

## 12. 测试与验收

### 12.1 单元测试

`test_memory_skeleton.py`：§4.3 每行一个用例，其中两条"不同"必须断言不相等。

`test_memory_ledger.py`
- 同一记录追加两次只返回一次新增；保存后恢复，再形成一次，统计不变（幂等）。
- `CsRank(x)` 与 `x` 的指纹相同；`Neg(x)` 与 `x` 相同；`x` 与 `x + noise` 不同。
- 历史 redundant 且锚因子已不在库 → precheck 不跳过。

`test_memory_layered.py`
- 同一骨架 6 个窗口变体 → n_eff = 1，不足以升 confirmed。
- 3 个不同骨架、各 1 次入库 → confirmed。
- 锚因子被替换后簇为 needs_recheck，不出现在 FORBIDDEN 主列表；成员再次评估后按新结果归簇。
- RECOMMENDED 每条同时含 supported by 与 boundary 字段。

`test_hypothesis.py`
- §8.4 例子：Δ=+0.003 → inconclusive。
- Δ=+0.012，区间不含 0 → supported；方向相反 → contradicted。
- 共同时点 < 60 → inconclusive；variant 实现失败 → untestable。
- base 入库、variant 低 IC，comparison 为 greater → 假设 contradicted 或 inconclusive，绝不 supported。
- 连续 3 次 inconclusive 不触发冷却；2 次 contradicted 触发。

### 12.2 回放验收（不需要 LLM）

用一次 mock 运行的 `memory_ledger.jsonl` 回放：

- 每条 P_succ/P_fail/I 陈述都能沿 support 追溯到账本记录（追溯率 100%）。
- 回放两遍（模拟恢复），统计完全相同。
- 不存在 "Novel:" / "HighCorr:" 条目；不存在不带两侧计数的比率类洞察。
- 用回放结果校准 §6.2 阈值，记录在运行清单。

### 12.3 闭环与消融验收

**闭环（第一项必须通过，否则记忆没有起作用）**

- 在同一运行中，被标为 saturated 的族，其后 5 轮内新候选落入该族的比例相对标记前下降。
- 进入 "Already tried" 的公式，在后续轮次的重复提交率下降。
- prompt 中的推荐族，在下一轮被生成的比例高于未推荐族。

**消融**（同 LLM、同数据、同 scope、同候选评估预算，各 3 个种子）

- A：`paper_rules`（当前规则版）；B：`layered`；C：`layered` + `hypothesis_groups`；D：`none`（无记忆）。
- 指标：
  - 精确重复率（规范化哈希 + 指纹）；
  - 单位评估预算的入库数；
  - 冗余拒绝率随轮次的变化；
  - 冻结后因子库在测试期的 IC/ICIR；
  - 每族的平均尝试数（过高说明在爬山）。
- 只把超出种子间波动的差异写成结论；测试期只在方案冻结后看一次。B 稳定优于 A，才把 `layered` 设为 paper 模式默认（§0.2）。

### 12.4 分期

| 期 | 交付 | 验收 |
|---|---|---|
| 1 | L0 账本（含 Scope、指纹、幂等）、L1 骨架 | 12.1 skeleton 与 ledger 测试 |
| 2 | L2 族统计、锚簇、状态机、五行渲染 | 12.1 layered 测试 + 12.2 回放 |
| 3 | 闭环检查与 A/B/D 消融 | 12.3 闭环三项 |
| 4 | 假设层（输出、配对判定、冷却） | 12.1 hypothesis 测试；真实 LLM 3 轮试跑解析成功率 ≥ 90%；消融 C |
| 5 | LLM 命名（可选） | 检查拒绝率与人工抽查 |
