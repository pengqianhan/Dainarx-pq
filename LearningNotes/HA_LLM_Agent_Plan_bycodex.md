# Hybrid Automaton LLM Agent 端到端方案（smolagent + SR-Scientist 思路）

> 目标：从 `npz`（主）与图像（辅）自动推理 Hybrid Automaton（HA）并输出与本仓库模拟器兼容的 JSON；移除对 `order/need_reset/self_loop/kernel/other_items` 的手写依赖，统一由 Agent 自动推断并写入 JSON 的 `meta`。

## 1. 设计原则
- 数据优先：模式发现、守卫/重置的主要证据来自时序（导数不连续、事件、局部残差），图像仅做弱先验。
- 工具增强的 LLM：LLM 负责“结构选择与搜索”，数值拟合与验证交由工具执行；形成 Propose → Execute → Evaluate → Refine 的闭环。
- 受限 DSL：使用受限的 ODE/Guard/Reset DSL，LLM 仅在 DSL 内组合，避免任意代码执行风险。
- MDL/可验证：以 MDL/BIC 与仿真误差为总评，Beam 搜索与自一致性裁剪，闭环验证并可自我修复。
- 与现有仓库无缝集成：输出 JSON 与 `HybridAutomata_simulate.py`、`src/DE.py`/`src/DE_System.py` 兼容，结果写入 `result/`。

## 2. 总体流程
1) 载入 `*.npz`（t, X[, U]）与可选图像；
2) 估计导数、构造特征与事件指示；
3) 变化点检测 + 聚类，选择模式数 K 并生成段标签；
4) LLM 给出项库（lib_config）建议；
5) 各模式用 SINDy（可辅以小预算符号回归）拟合方程；
6) 学习守卫与重置；
7) 组装 HA JSON；
8) 仿真评分，若超阈则迭代修正（K、项库、守卫/重置）。

## 3. 与 SR-Scientist 思路对齐
- 程序合成范式：LLM 生成结构假设（K、项库、guard/reset 形式），工具验证评分，返回失败归因，LLM 反思修正。
- 搜索策略：Beam Search/多温度采样 + 自一致性（取共识项），以 MDL/误差裁剪。
- DSL + 安全执行：受限符号集合与表达式模板，计算部分由工具解释与矢量化执行。

## 4. 架构（Hugging Face smolagent）
- Planner（主 Agent）：调度工具、保管搜索状态、决定下一步。
- Critic（可选）：阅读评分与失败归因，提出高层修正建议（增加/修剪项、调整 K/阈值）。
- Tools（原子能力）：以函数工具形式暴露，所有数值计算均在工具中完成。

### 4.1 工具清单与接口
- 数据与特征
  - `load_npz(path) -> {t, X[, U]}`
  - `estimate_derivatives(t, X, method, params) -> {Xdot, Xddot}`（Savitzky–Golay/TVReg，窗口与阶数网格搜索 + AIC/验证残差选）
  - `compute_features(X, Xdot, Xddot) -> features`（[x, ẋ, ẍ, 局部能量 H, 滑窗残差, 事件指示] 标准化）
- 分段与聚类
  - `detect_changepoints(features, strategy) -> candidates`（阈值 + KMeans/HDBSCAN + 动态规划平滑）
  - `select_num_modes(candidates, scores) -> K`（Silhouette + MDL/BIC）
  - `assign_modes(features, K) -> labels`（抖动平滑）
- 结构与方程
  - `propose_library(hint) -> lib_config`（由 LLM 决策：poly_deg、包含 {x, x^3, ẋ, sin/cos, 交叉项, 非光滑项}）
  - `sindy_fit(X, Xdot, lib_config, cv) -> {terms, coeffs, scores}`（STRidge/LassoCV，返回稀疏项与 CV 分数）
  - `sr_symbolic_search(X, Xdot, grammar, budget) -> best_exprs`（可选，小预算符号回归补充缺失非线性）
- 守卫与重置
  - `learn_guard(boundary_samples, forms=['linear','quad']) -> guard_expr, dir, score`
  - `learn_reset(prepost_pairs, forms=['linear','affine','componentwise']) -> reset_map, score`
- 组装与仿真
  - `assemble_ha_json(modes, edges, meta) -> json_obj`
  - `simulate_and_score(json_obj, t0, x0, t_ref, x_ref) -> metrics`（RMSE、事件 MAE、切换 F1、Fréchet）
- 图像辅助（可选）
  - `describe_plot(image_path) -> textual_hints`（几何/拓扑提示：地面直线、轨道形状、对称性）
  - `image_event_density(image_path) -> rough_event_rate`（极值密度近似事件率；弱先验）

备注：工具实现仅依赖 `numpy / scikit-learn / matplotlib`（图像可选 `PIL`），与仓库依赖一致。

### 4.2 DSL（受限表达）
- ODE（每模式）：`dx/dt = Σ c_k φ_k(x, ẋ, t)`；库函数集合由 `lib_config` 开关：
  - `1, x_i, x_i^2, x_i^3, x_i x_j, ẋ_i, sin(ax_i), cos(ax_i), sin(ωt), sign(ẋ_i), abs(x_i)`
  - 二阶系统用状态扩增成一阶（如 Duffing：`ẋ=v, v̇=...`）。
- Guard：`g(x) = 0, dir ∈ {>, <}`（入射方向，例：`ẏ<0`）；形式：线性 `a^T x + b` 或二次 `x^T Q x + b^T x + c`。
- Reset：`x+ = A x- + b`（线性/仿射/分量映射，如 `v+ = -c v-`）。

### 4.3 搜索与迭代（SR-Scientist 风格）
- 外层 Beam：宽度 B=3–5，迭代 T=3–5。
  - 生成候选（K 与 lib_config），分段→拟合→守卫/重置→组装 JSON→仿真评分→Top-k 入下一轮。
- 评分函数（越小越好）：`Score = α·RMSE_track + β·EventMAE + γ·SwitchF1 + λ·MDL`。
- 自一致性：不同温度/不同启发生成多份结构建议，保留共识项（如 Duffing 的 `x^3`）。
- 失败归因：切换漏检→增加阈值/简化 guard；段内残差高→提高多项式阶/加入阻尼或三角项；MDL 高→压缩项/降低 K。

## 5. 数据处理与模式发现（Duffing vs Ball）
- 导数稳健估计：默认 Savitzky–Golay，窗口与阶数网格搜索 + AIC/验证残差；噪声大时 TVReg 备选。
- 事件指示器：‖Δẋ‖、‖Δẍ‖ 峰值；零交叉/极值密度；滑窗拟合残差突增。
- 聚类和平滑：在 `[x, ẋ, ẍ, H, 残差]` 空间 KMeans/HDBSCAN，Silhouette 与 MDL 选 K；动态规划最小化切换成本。
- Duffing：通常 K=1；图像仅用于提升三次项、阻尼项优先级以及外激励猜测。
- Bouncing Ball：K=2（飞行/碰撞）；guard `y=0 ∧ ẏ<0`，reset `ẏ+ = −c ẏ−`；若存在接触/摩擦，可扩展为 K≥3。

## 6. 方程结构与参数
- LLM 给出项库建议（`lib_config`：poly_deg、包含/排除项、是否 trig、是否非光滑项）。
- SINDy 拟合：STRidge/LassoCV；交叉验证分数 + 稀疏系数稳定性过滤项。
- 符号回归（可选，小预算）：发现如 `x·|x|`、`x·ẋ` 等 SINDy 难以捕获的非线性，再回流到 SINDy 精确化参数。
- 选择准则：MDL/BIC + 验证误差；偏向少而稳的可解释项。

## 7. 守卫与重置学习
- Guard：对模式切换边界正/负侧样本拟合线性与二次超平面（RANSAC 抗离群）；用速度符号决定入射方向；以 MDL/精度选择。
- Reset：聚合切换时刻的 pre/post 状态，拟合线性/仿射；若呈弹性碰撞型，约束为 `v+ = -c v-` 并估计 `c`；能量一致性作为 sanity check。
- Self-loop：模式内默认 self-loop；必要时加入 invariant（如 `y>0`）。

## 8. 图像利用（可选，弱先验）
- 描述：相图是否双稳势/对称，是否存在明显接触直线（Hough 直线），撞击点是否沿直线聚集。
- 用途：仅影响 `K` 的初始猜测与 `lib_config` 优先级；实际分段/拟合由时序决定。

### 8.1 两种图像情形的处理策略（有/无明显切换）
- 共同原则
  - 图像只做“弱先验”，不直接设定 `K` 或标签；所有候选必须通过数值评分闭环（simulate_and_score）验证。
  - 若图像先验与序列证据冲突，默认以时序证据为准；保留 `K` 与 `K+1` 两份候选，比较 MDL 与仿真误差后再定。

- 情形 A：图像有明显模式切换（如 Bouncing Ball）
  - 先验与工具
    - `image_event_density` 高：优先级上调 `K≥2` 的候选；`select_num_modes` 接受 `image_prior.rough_event_rate` 作为排序信号。
    - `describe_plot` 检测到“地面直线/撞击条纹”：对 Guard 形式施加强偏置，优先线性 `y=0` 且方向性 `ẏ<0`（入射）。
  - 分段与守卫/重置
    - 分段阈值更敏感，`detect_changepoints` 允许更高的候选密度；`assign_modes` 使用动态规划平滑避免抖动。
    - Guard 首选线性（必要时回退/升阶到二次），强制方向约束；Reset 首选分量仿射，并带物理先验 `v+ = -c v-`，用 RANSAC 估计 `c`。
  - 评分与选择
    - 在 `Score = α·RMSE + β·EventMAE + γ·(1-F1) + λ·MDL` 中，提高事件相关权重（例如 β≈0.35, γ≈0.25），确保切换时间与次数对齐；
      若 `K` 与 `K+1` 评分相近，优先更低 MDL（更简洁）。

- 情形 B：图像无明显模式切换（如 Duffing）
  - 先验与工具
    - `image_event_density` 低：`select_num_modes` 提升 `K=1` 候选优先级，并提高对 `K>1` 的 MDL 惩罚。
    - `describe_plot` 若提示双井/对称形态，仅作为项库提示（`x^3`、阻尼、可能存在 `cos(ω t)`），不触发分段。
  - 分段与方程
    - 默认 `K=1`（经 Silhouette/MDL 支持），`assign_modes` 加强平滑避免伪切换；模式内保留 self-loop，无 Guard/Reset。
    - SINDy 库优先 `{x, x^3, ẋ}`，必要时加入 `sin(ω t)`；使用 CV/MDL 控制复杂度；可用小预算符号回归补齐难项。
  - 评分与选择
    - 将事件权重降到较低（例如 β≈0.1, γ≈0.0–0.05），以轨迹 RMSE 与 MDL 为主；
      若误将数据切成多模但 SwitchF1 低/MDL 高，则合并回 `K=1`。

### 8.2 冲突与自修复（Planner 策略）
- 证据融合：`select_num_modes` 接收 `image_prior`（粗事件率、是否检测到边界直线）与时序分数（Silhouette、MDL）共同排序候选 `K∈{1..Kmax}`。
- 失败归因到动作：
  - 切换漏检 → 提高分段灵敏度、简化 Guard 形式、放宽事件对齐容差；
  - 段内残差高 → 扩大项库（如加入 `x^3`、阻尼项、三角项）、提高正则；
  - 过分段/过拟合 → 提升 MDL 权重、收紧阈值、合并相邻段。
- 保留双解：当图像强烈暗示 `K≥2` 而时序倾向 `K=1` 时，同时输出两份 JSON 与评分，默认选择更低 MDL，记录备选到 `result/<run>/alt_candidates/`。

### 8.3 实现落点（工具/Prompt 接口）
- 工具
  - `image_event_density(image) -> rough_event_rate ∈ [0,1]`；`describe_plot(image) -> {has_floor_line, symmetry, periodicity_hint}`。
  - `select_num_modes(..., image_prior)`：基于 `rough_event_rate` 与直线检测结果对候选 `K` 排序并调参阈值。
  - `propose_library(..., textual_hints)`：若 `symmetry/双井`→提升 `x^3`；若 `periodicity_hint`→增加 `sin(ω t)` 候选。
- Prompt（planner.md/critic.md）
  - 明确规则：
    - “若 image_prior 显示明显切换，则优先尝试 `K≥2`，Guard 优先线性+方向性，重置优先 `v+ = -c v-`。”
    - “若 image_prior 显示无切换，则优先 `K=1`，增加对 `K>1` 的复杂度惩罚，仅将图像用于项库提示。”
  - Critic 动作集合：（increase/decrease）`seg_thresholds`、`mdl_weight`、`lib_config.poly_deg/terms`、`event_tolerance`、`guard_form`。

## 9. JSON 输出规范（与模拟器兼容）
- modes：每个模式含变量方程列表；
- edges：源/目标、guard（表达式 + 方向）、reset（映射表达式）；
- meta：自动推断的 `order/need_reset/self_loop/kernel` 与评分、随机种子等。

示例（Duffing 单模式）：
```json
{
  "modes": [
    {
      "name": "m1",
      "equations": [
        {"var": "x", "rhs": "v"},
        {"var": "v", "rhs": "-delta*v - alpha*x - beta*x^3"}
      ]
    }
  ],
  "edges": [
    {"source": "m1", "target": "m1", "guard": {"expr": "false"}, "reset": {"expr": "identity"}}
  ],
  "meta": {"order": 3, "need_reset": false, "self_loop": true, "kernel": "poly(3)", "other_items": {"scores": {}}}
}
```

示例（Ball 双模式，飞行 m1 → 撞击 m2）：
```json
{
  "modes": [
    {"name": "m1", "equations": [
      {"var": "y", "rhs": "v"},
      {"var": "v", "rhs": "-g"}
    ]},
    {"name": "m2", "equations": [
      {"var": "y", "rhs": "v"},
      {"var": "v", "rhs": "-g"}
    ]}
  ],
  "edges": [
    {"source": "m1", "target": "m2", "guard": {"expr": "y", "dir": "<"}, "reset": {"expr": "v := -c * v"}},
    {"source": "m2", "target": "m1", "guard": {"expr": "y", "dir": ">"}, "reset": {"expr": "identity"}}
  ],
  "meta": {"order": 2, "need_reset": true, "self_loop": true, "kernel": "poly(1)", "other_items": {"scores": {}}}
}
```

## 10. 评分与模型选择
- 段内动力学：R2、RMSE、滚动一步预测误差；
- 切换：事件时间 MAE、切换 F1（含假阳/假阴）；
- 整体仿真：轨迹 RMSE、Fréchet 距离、稳态/能量一致性；
- 复杂度：MDL/BIC（模式数、项数、guard/reset 复杂度）。
- 总分：`Score = α·RMSE + β·EventMAE + γ·(1-F1) + λ·MDL`（可按任务调整权重）。

## 11. 失败保护与人工介入
- K 不确定：同时输出 K、K+1 两份 JSON 与评分，默认选择更低 MDL；
- Guard/Reset 不稳：回退线性形式并强制方向约束；
- 高噪声：切换到 TVReg 导数与更强 RANSAC；
- 允许最小提示：是否存在外激励/控制量、最大多项式阶、最大模式数。

## 12. 与仓库集成
- 目录与模块（建议新增）：
  - `src/agent_ha_smol.py`：smolagent 入口与 Planner/Critic 调度
  - `src/tools/features.py`：导数、特征、事件
  - `src/tools/segmentation.py`：变化点/聚类/平滑
  - `src/tools/sindy.py`：项库与 SINDy 拟合封装
  - `src/tools/guards.py`：守卫/重置学习
  - `src/tools/assemble.py`：JSON 组装与仿真评分
  - `prompts/planner.md`, `prompts/critic.md`：SR-Scientist 风格模板（严格 JSON 指令）
- 入口扩展：
  - `python main.py automata/example.json`（保持原有路径）
  - 新增：`python main.py --agent ha_llm --npz data/...npz [--image data_plot/...png] --out result/run_x`
  - Agent 路径：npz → Agent → `automata/<run_name>_llm.json` → 调用现有评估/仿真 → 输出指标与图。
- 依赖：基线依赖不变（`numpy, scikit-learn, matplotlib, networkx`）；smolagent 为额外依赖（如需离线，可先以简单调度器替代）。

## 13. 最小可行版本（里程碑）
- A：npz → 导数/特征 → 单模式 SINDy → JSON → 仿真评分（Duffing）。
- B：变化点/分段 + 守卫/重置学习 → 双模式（Ball），闭环验证。
- C：加入符号回归补充项、Beam 搜索与 Critic；多候选模型与 MDL 选择。
- D：图像弱先验、报告与可视化（对比曲线、事件对齐图、段标注）。

## 14. 复现实验建议
- Duffing：
  - 期望 K=1；库含 `{x, x^3, ẋ}`，可探测 `cos(ω t)`；
  - 输出 `automata/duffing_llm.json`，`result/duffing_llm/*` 含对比图与指标；
  - 指标阈值：轨迹 RMSE < 原始噪声水平、MDL 低、事件 MAE 不适用（无显式切换）。
- Bouncing Ball：
  - 期望 K=2；guard `y=0 ∧ ẏ<0`，reset `ẏ+ = −c ẏ−`；
  - 输出 `automata/ball_llm.json`；
  - 指标阈值：切换 F1 > 0.9，事件 MAE < dt 的 2–3 倍，RMSE 低且能量耗散率一致。

## 15. 风险与缓解
- 模式不可分（如 Duffing）：K=1 回退，保留 self-loop；
- 噪声高：更强正则与 RANSAC、TVReg 导数、段内加权拟合；
- 过拟合：MDL 惩罚、交叉验证、共识项优先；
- 计算预算：限制 Beam 宽度与迭代轮次，缓存子结果。

## 16. 关键函数签名与伪代码

### 16.1 Python 函数签名（示例）
```python
def estimate_derivatives(t: np.ndarray, X: np.ndarray, method: str = "savgol", params: Dict | None = None) -> Dict[str, np.ndarray]:
    ...

def sindy_fit(X: np.ndarray, Xdot: np.ndarray, lib_config: Dict, cv: int = 5) -> Dict:
    """return {"terms": List[str], "coeffs": np.ndarray, "scores": {"r2": float, "rmse": float, "mdl": float}}"""
    ...

def learn_guard(boundary: np.ndarray, labels: np.ndarray, forms: List[str]) -> Tuple[Dict, float]:
    ...

def learn_reset(pre: np.ndarray, post: np.ndarray, forms: List[str]) -> Tuple[Dict, float]:
    ...

def simulate_and_score(ha_json: Dict, t_ref: np.ndarray, x_ref: np.ndarray) -> Dict[str, float]:
    ...
```

### 16.2 smolagent 工具与 Planner 伪代码
```python
from smolagent import Tool, Agent

class SindyFitTool(Tool):
    name = "sindy_fit"
    description = "Fit sparse dynamics with given library config"
    inputs = {"X": "array", "Xdot": "array", "lib_config": "json"}
    output_schema = {"terms": list, "coeffs": list, "scores": dict}
    def __call__(self, X, Xdot, lib_config):
        return sindy_fit(X, Xdot, lib_config, cv=5)

planner = Agent(
    tools=[LoadNPZ(), DerivativeTool(), FeatureTool(), ChangepointTool(), SindyFitTool(), GuardTool(), ResetTool(), AssembleTool(), SimScoreTool()],
    system_prompt=open("prompts/planner.md").read()
)

state = planner.run({"npz": path, "image": maybe_image})  # 多轮调用，遵循严格 JSON 指令
```

### 16.3 Planner 循环（高层）
1) 调 `load_npz` / `estimate_derivatives` / `compute_features`；
2) 调 `detect_changepoints`，`select_num_modes`，`assign_modes` 得到 K、labels；
3) 结合图像提示与统计摘要，请 LLM 产出 `lib_config`；
4) 各模式调用 `sindy_fit`（必要时 `sr_symbolic_search`）；
5) `learn_guard` / `learn_reset`；
6) `assemble_ha_json` 并 `simulate_and_score`；
7) 若分数不达标：Critic 给出修正建议 → 返回第 2–6 步迭代；
8) 输出最佳 JSON 与评分、可视化结果。

## 17. 复现与随机性
- 固定 `np.random.seed` 与 SINDy/聚类初始化；
- 将随机种子、超参数与版本信息写入 JSON `meta`；
- 所有可视化、CSV 指标写入 `result/<run_name>/`，便于 PR 前检查与回归。

---

附注：若离线/无网络环境无法安装 smolagent，可用简单的 Python 调度器先替代（函数驱动的 ReAct 循环与严格 JSON 契约），待网络可用时再切换到 smolagent。
