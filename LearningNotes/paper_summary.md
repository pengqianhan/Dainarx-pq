
### 核心要素分析

*   **根本问题**
    传统上，从数据中识别“混合系统”（即会在不同工作模式间切换的系统）依赖于检测其信号的“导数突变”。这种方法不仅对噪声敏感，更关键的是需要人为设定一个难以把握的“突变阈值”——阈值太高会漏掉真实的模式切换，太低则会引入大量虚假的切换点，尤其对于复杂的非线性系统，此矛盾几乎无解。

*   **切入视角**
    本文提出了一个根本性的视角转变：模式切换的本质特征并非信号的“导数剧变”，而是系统背后“动态规律（dynamic model）的不再适用”。即，只要系统遵循同一套物理或控制法则，无论其信号如何变化，都应该能被同一个数学模型完美拟合；反之，当这个模型开始失效时，就意味着系统切换到了一个新的内在模式。

*   **关键方法**
    作者将上述视角落地为一种统一、无阈值的核心机制：**将非线性自回归模型（NARX）的拟合能力作为“试金石”**。具体而言，通过一个滑动窗口持续尝试用单一 NARX 模型来拟合局部数据段：
    1.  **分割 (Segmentation)：** 如果一段数据能被一个 NARX 模型很好地拟合，则它属于同一个模式；当拟合误差首次超过一个极小的、机器级的容忍度时，即标志着一个模式切换点。
    2.  **聚类 (Clustering)：** 所有能被*同一个* NARX 模型参数拟合的数据段，都被归为同一类，即对应系统的同一种工作模式。

*   **核心发现**
    通过这种"以模型拟合度量内在规律"的方法，可以在无需任何人工阈值的情况下，精确、鲁棒地从观测数据中反推出高阶、非线性混合系统的完整结构（包括每种模式下的动态方程、模式切换的边界条件以及切换时的状态重置规则），其精度和适用范围均显著超越了依赖导数突变的传统方法。

---

## 技术实现详解

### 1. NARX 模型的数学形式

NARX (Nonlinear AutoRegressive with eXogenous inputs) 模型是本方法的核心工具。对于第 i 个状态变量，其 NARX 模型形式为：

```
x_i[t] = Σ(a_j · x_i[t-j]) + Σ(b_k · u_k[t]) + Σ(c_l · φ_l(x, u)) + bias
         j=1 to order      k=1 to input_num    l=1 to nonlinear_terms
```

其中：
- `x_i[t]` 是第 i 个状态变量在时刻 t 的值
- `a_j` 是自回归系数，捕捉历史状态的影响
- `u_k[t]` 是外部输入信号
- `φ_l(x, u)` 是可选的非线性项（如 x₁·x₂, sin(x₁) 等）
- `order` 是模型阶数，决定了历史依赖的时间深度

**代码实现** (src/DEConfig.py:147-159)：
```python
def append_data(self, matrix_list, b_list, data: np.array, input_data):
    """构造 NARX 模型的最小二乘回归矩阵"""
    data = np.array(data)
    input_data = np.array(input_data)
    for i in range(len(data[0]) - self.order):
        # 提取滞后项 x[t-1], x[t-2], ..., x[t-order]
        if i == 0:
            this_line = data[:, (self.order - 1)::-1]
            this_line_input = input_data[:, self.order::-1]
        else:
            this_line = data[:, (self.order + i - 1):(i - 1):-1]
            this_line_input = input_data[:, (self.order + i):(i - 1):-1]
        for idx in range(len(this_line)):
            # 构造特征向量 [x[t-1], x[t-2], ..., u[t], bias, nonlinear_terms]
            matrix_list[idx].append(self.get_items(this_line, this_line_input, idx))
            b_list[idx].append(data[idx][i + self.order])  # 目标值 x[t]
```

#### 1.1 非线性项的添加过程

完整的非线性项工作流程

  步骤1：在初始化时解析非线性表达式
  ```python
class FeatureExtractor:
    def __init__(self, var_num: int, input_num: int, order: int, dt: float,
               need_bias: bool = False, minus: bool = False, other_items: str = ''):
      self.var_num = var_num
      self.order = order
      self.dt = dt
      self.input_num = input_num
      self.minus = minus
      self.need_bias = need_bias
      # ⭐ 关键：将字符串表达式转换为可调用的函数
      self.fun_list, self.fun_order = FeatureExtractor.analyticalExpression(
          other_items,  # ← 例如: "x[1]*x[2]; x[1]**2; sin(x[1])"
          var_num,
          order
      )
  ```
  步骤2：analyticalExpression 将字符串转换为 lambda 函数
  ```python
  @staticmethod
  def analyticalExpression(expr_list: str, var_num, order):
      """将字符串表达式解析为可执行的 lambda 函数"""
      res = [[] for _ in range(var_num)]
      res_order = [[] for _ in range(var_num)]

      expr_list = expr_list.split(';')  # 分割多个表达式

      for idx in range(var_num):
          expr = FeatureExtractor.extractValidExpression(expr_list, idx)
          expr = FeatureExtractor.unfoldDigit(expr, order)
          expr = FeatureExtractor.unfoldItem(expr, idx, var_num)

          for s in expr:
              # ⭐ 核心：将字符串转换为 lambda 函数
              res[idx].append(eval('lambda x: ' + s))
              # 例如："x[0][1]*x[1][1]" → lambda x: x[0][1]*x[1][1]
              res_order[idx].append(FeatureExtractor.findMaxorder(s) + 1)

      return res, res_order
  ```
  步骤3：get_items 调用这些 lambda 函数
  ```python
  def get_items(self, data, input_data, idx, max_order=None):
      res = []

      # ... 添加线性滞后项 ...

      # ⭐ 这里执行非线性函数！
      for (fun, order) in zip(self.fun_list[idx], self.fun_order[idx]):
          if order > max_order:
              res.append(0.)
          else:
              res.append(fun(data))  # 调用 lambda 函数
              # 例如：fun = lambda x: x[0][1] * x[1][1]
              #      data = [[...], [...], ...]
              #      结果：data[0][1] * data[1][1]

      # ... 添加输入项和偏置项 ...

      return res
  ```
  具体示例说明
  
  假设我们要为 Duffing 振子添加非线性项：
```python
  # 配置文件中设置
  config = {
      "order": 3,
      "other_items": "x[1]**3; x[1]*x[2]"  # ← 定义非线性项
  }

  # 初始化时
  get_feature = FeatureExtractor(
      var_num=2,      # 两个状态变量 x1, x2
      input_num=1,
      order=3,
      dt=0.01,
      other_items="x[1]**3; x[1]*x[2]"
  )

  # 此时 self.fun_list 包含：
  # fun_list[0] = [lambda x: x[0][0]**3, lambda x: x[0][0]*x[1][0]]
  # fun_list[1] = [lambda x: x[1][0]**3, lambda x: x[1][0]*x[2][0]]
  ```
  在拟合时，对于第一个变量（idx=0），特征向量构成为：
  ```python
  特征向量 = [
      x1[t-1],           # 线性滞后项
      x1[t-2],
      x1[t-3],
      (x1[t-1])³,        # ← 非线性项 1: x[1]**3
      x1[t-1] * x2[t-1], # ← 非线性项 2: x[1]*x[2]
      u[t],              # 输入项
      1                  # 偏置项
  ]
  ```
  最终的 NARX 模型就变成：
  ```python
  x1[t] = a1·x1[t-1] + a2·x1[t-2] + a3·x1[t-3]
          + a4·(x1[t-1])³           # ← 非线性项
          + a5·x1[t-1]·x2[t-1]      # ← 交叉非线性项
          + b1·u[t]
          + bias
  ```
  ★ Insight ─────────────────────────────────────
  1. 非线性项的可扩展性：通过字符串表达式 + eval() 的设计，用户可以灵活添加任意数学表达式（如三角函数、指数、多项式等），而不需要修改核心代码。
  2. "惰性求值"设计：fun_list 存储的是 lambda 函数而不是具体值，只有在调用 get_items() 时才实际计算，避免了不必要的内存开销。
  3. 这种设计体现了论文的核心思想：NARX 模型不仅仅是线性自回归，通过添加非线性项，它可以逼近任意复杂的非线性动力学系统（类似于神经网络的通用逼近定理）。
  ─────────────────────────────────────────────────

  代码流程总结
  ```markdown

  用户配置: other_items = "x[1]**3; sin(x[1])"
      ↓
  [初始化阶段]
  analyticalExpression() 解析字符串
      ↓
  生成 fun_list = [lambda x: x[0][0]**3, lambda x: sin(x[0][0])]
      ↓
  [拟合阶段]
  append_data() 构造回归矩阵
      ↓
      ↓
  get_items() 依次添加：
    - 线性滞后项
    - 非线性项 ← 调用 fun(data)
    - 输入项
    - 偏置项
      ↓
  返回完整特征向量用于最小二乘拟合
  ```
**最小二乘拟合** (src/DEConfig.py:124-139)：
```python
def work_normal(self, data, input_data, is_list: bool):
    """使用最小二乘法拟合 NARX 模型"""
    res = []
    err = []
    var_num = len(data) if not is_list else len(data[0])
    matrix_list = [[] for _ in range(var_num)]
    b_list = [[] for _ in range(var_num)]

    # 构造回归矩阵 A 和目标向量 b
    if is_list:
        for block, block_input in zip(data, input_data):
            self.append_data(matrix_list, b_list, block, block_input)
    else:
        self.append_data(matrix_list, b_list, data, input_data)

    # 对每个变量求解 min ||Ax - b||²
    for a, b in zip(matrix_list, b_list):
        x = np.linalg.lstsq(a, b, rcond=None)[0]  # 最小二乘解
        res.append(x)
        err.append(max(np.abs((a @ x) - b)))  # 最大拟合残差
    return res, err, [self.order for _ in range(self.var_num)]
```

### 2. 机器级容忍度的计算

这是"无阈值"方法的关键——不是完全没有阈值，而是使用一个**与数据尺度自适应的极小阈值**，避免人为调参。

**代码实现** (src/DEConfig.py:77-78)：
```python
def get_eps(self, data):
    """计算机器级容忍度阈值"""
    return 1e-6 * self.dt * np.max(data)
```

这个公式的设计思想：
- `1e-6` 是一个接近浮点运算精度的极小常数
- `dt` 是采样时间间隔，将误差归一化到单位时间
- `np.max(data)` 使阈值与数据量级成正比，适应不同尺度的系统

**举例**：如果数据最大值为 100，采样间隔 dt=0.01，则 `eps = 1e-6 * 0.01 * 100 = 1e-6`，这是一个极小的拟合误差要求。

### 3. 滑动窗口分割算法

分割算法通过滑动窗口检测模式切换点，核心思想是：**当单一 NARX 模型无法继续拟合新数据时，标记为切换点**。

**算法流程** (src/ChangePoints.py:16-49)：
```python
def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=None):
    """
    参数：
        data: 状态变量时间序列 (N变量 × M时间点)
        input_data: 输入信号时间序列
        get_feature: NARX 拟合函数
        w: 滑动窗口大小
        merge_th: 合并相邻切换点的阈值
    """
    change_points = []
    pos = 0
    last = None
    tail_len = 0
    if merge_th is None:
        merge_th = w

    # 计算机器级容忍度
    eps = get_feature.get_eps(data)

    # 滑动窗口遍历时间序列
    while pos + w < data.shape[1]:
        # 对当前窗口拟合 NARX 模型
        feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)],
                                                   input_data[:, pos:(pos + w)])

        if last is not None:
            # 核心判断：如果拟合误差超过容忍度，标记切换点
            if (max(now_err) > eps) and tail_len == 0:
                change_points.append(pos + w - 1)
                tail_len = w  # 避免在同一区域重复检测
            tail_len = max(tail_len - 1, 0)

        last = fit_order
        pos += 1

    # 合并过于接近的切换点
    res = mergeChangePoints(change_points, merge_th)
    res.append(data.shape[1])
    res.insert(0, 0)
    return res
```

**关键设计**：
- **tail_len 机制**：检测到切换点后，跳过后续 w 个位置，避免在同一切换边界重复标记
- **mergeChangePoints**：如果两个切换点间隔小于 merge_th，只保留第一个，消除噪声导致的虚假检测

**可视化理解**：
```
时间序列: |====模式A====|****|====模式B====|****|====模式C====|
           ↑           ↑   ↑           ↑   ↑
         窗口位置     拟合  切换点1      拟合  切换点2
                    良好  err>eps     失败  err>eps
```

### 4. 动态阈值校准

虽然分割阶段使用了机器级阈值，但在聚类阶段需要一个更宽松的阈值，用于判断"两个数据段是否属于同一模式"。这个阈值通过**训练数据自动校准**。

**代码实现** (src/CurveSlice.py:32-54)：
```python
@staticmethod
def fit_threshold_one(get_feature, data1, data2):
    """比较两个相邻数据段，校准 FitErrorThreshold"""
    feature1 = data1.feature
    feature2 = data2.feature

    # 尝试用单一 NARX 模型同时拟合这两个数据段
    _, err, fit_order = get_feature([data1.data, data2.data],
                                  [data1.input_data, data2.input_data], is_list=True)

    # 如果拟合阶数没有增加，说明两段可能属于同一模式
    if fit_order <= max(data1.fit_order, data2.fit_order):
        # 将阈值设为拟合误差的 0.1 倍（容忍度比例）
        Slice.FitErrorThreshold = min(Slice.FitErrorThreshold,
                                      max(err) * Slice.ToleranceRatio)
        Slice.FitErrorThreshold = max(Slice.FitErrorThreshold, 1e-6)

    # 同时校准基于参数距离的阈值
    idx = 0
    for v1, v2 in zip(feature1, feature2):
        relative_dis, dis = Slice.get_dis(v1, v2)
        if relative_dis > 1e-4:
            Slice.RelativeErrorThreshold[idx] = \
                min(Slice.RelativeErrorThreshold[idx], relative_dis * Slice.ToleranceRatio)
        if dis > 1e-4:
            Slice.AbsoluteErrorThreshold[idx] = \
                min(Slice.AbsoluteErrorThreshold[idx], max(dis * Slice.ToleranceRatio, 1e-6))
        idx += 1
    return True
```

**设计思想**：
- 遍历训练数据中所有相邻的数据段对
- 如果相邻段属于**不同模式**，它们的联合拟合误差就代表了"模式间差异"
- 将阈值设为这些差异的 10% (`ToleranceRatio = 0.1`)，既保证了区分度，又允许一定的噪声容忍

**示意图**：
```
段1(模式A)  段2(模式B)
  ↓           ↓
  合并拟合 → err = 0.05  →  FitErrorThreshold = 0.005 (10%)
```

### 5. 基于拟合能力的聚类算法

这是方法的核心创新：**不是比较参数向量的距离，而是测试"能否用同一组 NARX 参数拟合不同数据段"**。

**代码实现** (src/Clustering.py:4-61)：
```python
def clustering(data: list[Slice], self_loop=False):
    """
    基于 NARX 拟合能力的聚类算法

    核心思想：
        - 能被同一组 NARX 模型拟合的数据段 → 同一模式
        - 维护 mode_dict[mode_id] = [代表性数据段列表]
    """
    tot_mode = 1
    last_mode = None
    mode_dict = {}  # 存储每个模式的代表性数据段
    delay_list = []  # 暂时无法分类的歧义数据段

    for i in range(len(data)):
        data[i].idx = i
        if not data[i].valid:
            continue
        if data[i].isFront:
            last_mode = None

        if Slice.Method == 'fit':  # 默认方法：基于拟合能力
            fit_cnt = 0
            # 遍历已识别的所有模式
            for idx, val in mode_dict.items():
                # 测试：当前数据段能否被该模式的代表段拟合？
                if idx != last_mode and data[i].test_set(val):
                    data[i].mode = idx
                    fit_cnt += 1
                # 处理二阶段匹配的特殊情况
                elif len(val) == 2:
                    if data[i].test_set([val[0]]) or data[i].test_set([val[1]]):
                        delay_list.append(val[-1])
                        mode_dict[idx].pop()
                        fit_cnt = 2

            # 情况1：恰好匹配一个模式
            if fit_cnt == 1:
                if len(mode_dict[data[i].mode]) < 3:
                    mode_dict[data[i].mode].append(data[i])  # 添加为代表段
            # 情况2：匹配多个模式（歧义）
            elif fit_cnt > 1:
                delay_list.append(data[i])
                data[i].mode = -1

        # 情况3：不匹配任何现有模式 → 创建新模式
        if data[i].mode is None:
            data[i].mode = tot_mode
            mode_dict[tot_mode] = [data[i]]
            tot_mode += 1

        if not self_loop:
            last_mode = data[i].mode

    # 第二阶段：处理歧义数据段
    if Slice.Method == 'fit':
        for s in delay_list:
            for idx, val in mode_dict.items():
                # 利用时序信息：优先匹配前一段的模式
                if not s.isFront and data[s.idx - 1].mode == idx:
                    continue
                if s.test_set(val):
                    s.mode = idx
                    if len(val) < 3:
                        mode_dict[idx].append(s)
                    break
            # 仍无法匹配 → 创建新模式
            if s.mode is None or s.mode == -1:
                s.mode = tot_mode
                mode_dict[tot_mode] = [s]
                tot_mode += 1
```

**test_set 函数** (src/CurveSlice.py:90-104)：
```python
def test_set(self, other_list):
    """测试当前数据段能否被 other_list 中的段共同拟合"""
    data, input_data, other_fit_order = [], [], None

    # 收集参考段的数据
    for s in other_list:
        data.append(s.data)
        input_data.append(s.input_data)
        if other_fit_order is None:
            other_fit_order = copy.copy(s.fit_order)
        else:
            other_fit_order = min(other_fit_order, s.fit_order)

    # 尝试用单一 NARX 模型拟合当前段+所有参考段
    _, err, fit_order = self.get_feature([self.data] + data,
                                        [self.input_data] + input_data,
                                        is_list=True)

    # 判断条件：
    # 1. 拟合阶数不增加（模型复杂度不变）
    # 2. 拟合误差小于校准的阈值
    order_condition = True
    for i in range(len(fit_order)):
        order_condition = order_condition and \
                         fit_order[i] <= max(self.fit_order[i], other_fit_order[i])

    return order_condition and max(err) < Slice.FitErrorThreshold
```

**为什么这样设计有效？**

如果两个数据段来自同一个动力学模式，那么用单一 NARX 模型联合拟合它们应该：
1. **不需要增加模型阶数**：同一物理规律下，时间依赖深度应该一致
2. **拟合误差保持在容忍范围内**：同一动力学方程产生的数据应该能被统一描述

这种判断方式**直接基于物理意义**，而不是参数空间的几何距离。

**聚类过程示意**：
```
段1 段2 段3 段4 段5 段6
 ↓   ↓   ↓   ↓   ↓   ↓
[段1] → 模式1
[段2] test_set([段1]) → 通过 → 模式1
[段3] test_set([段1,段2]) → 失败 → 模式2
[段4] test_set([段1,段2]) → 通过 → 模式1
[段5] test_set([段3]) → 通过 → 模式2
[段6] test_set([段1,段2,段4]) → 失败 → test_set([段3,段5]) → 失败 → 模式3
```

### 6. 完整的识别流程

**主流程代码** (main.py:23-47)：
```python
def run(data_list, input_data, config, evaluation: Evaluation):
    """混合系统识别主流程"""
    input_data = np.array(input_data)

    # 1. 初始化 NARX 特征提取器
    get_feature = FeatureExtractor(len(data_list[0]), len(input_data[0]),
                                   order=config['order'], dt=config['dt'],
                                   minus=config['minus'],
                                   need_bias=config['need_bias'],
                                   other_items=config['other_items'])

    Slice.clear()
    slice_data = []
    chp_list = []

    # 2. 分割：对每条训练轨迹检测切换点
    for data, input_val in zip(data_list, input_data):
        change_points = find_change_point(data, input_val, get_feature,
                                        w=config['window_size'])
        chp_list.append(change_points)
        print("ChP:\t", change_points)
        slice_curve(slice_data, data, input_val, change_points, get_feature)

    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")

    # 3. 阈值校准
    Slice.Method = config['clustering_method']
    Slice.fit_threshold(slice_data)

    # 4. 聚类：识别模式
    clustering(slice_data, config['self_loop'])
    evaluation.recording_time("clustering")

    # 5. 学习守卫条件（模式切换规则）
    adj = guard_learning(slice_data, get_feature, config)
    evaluation.recording_time("guard_learning")

    # 6. 构建混合自动机
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    evaluation.submit(slice_data=slice_data)

    return sys, slice_data
```

**流程总结**：
```
输入数据
   ↓
[1] 滑动窗口 + NARX拟合 → 检测切换点 → 数据段列表
   ↓
[2] 动态校准阈值 ← 相邻段拟合误差统计
   ↓
[3] 基于拟合能力的聚类 → 模式识别
   ↓
[4] SVM学习守卫条件 → 切换规则
   ↓
[5] 构建混合自动机 ← 模式+切换规则+重置函数
   ↓
输出：可仿真的混合系统模型
```

### 7. 实验配置示例

**Duffing 振子配置** (automata/non_linear/duffing.json)：
```json
{
  "config": {
    "order": 3,                    // NARX 模型阶数
    "window_size": 10,             // 滑动窗口大小
    "clustering_method": "fit",    // 聚类方法："fit"（拟合能力）或 "dis"（距离）
    "need_bias": true,             // 是否包含偏置项
    "need_reset": false,           // 是否学习状态重置函数
    "kernel": "linear",            // SVM 核函数
    "svm_c": 1e6,                  // SVM 正则化参数
    "other_items": "",             // 自定义非线性项（如 "x[1]*x[2]"）
    "minus": false,                // 是否自动减少模型阶数
    "self_loop": false,            // 是否允许自环（同一模式内跳转）
    "class_weight": 1.0,           // SVM 类别权重
    "dt": 0.01,                    // 采样时间间隔
    "total_time": 10               // 仿真总时长
  }
}
```

**关键参数说明**：
- `order`：决定了 NARX 模型能捕捉多长的时间依赖（通常 2-5）
- `window_size`：窗口越大，对噪声越鲁棒，但可能错过短暂的模式
- `clustering_method = "fit"`：推荐设置，基于物理意义的拟合能力聚类
- `other_items`：可扩展性接口，用于添加领域知识的非线性项

### 方法公式化

**系统模式识别 = ¬ (单一 NARX 模型 拟合 数据段)** + **(模型参数集 ≈ 模式标识)**

这个公式表达了：
1.  模式的**切换点**通过判断“单一NARX模型无法拟合当前数据段”来确定（用逻辑非“¬”表示）。
2.  模式的**类别**通过“能够成功拟合某数据段的模型参数集”来定义和聚类。

### 最终双重总结

*   **一句话总结 (核心价值)**
    为了解决传统方法依赖脆弱的导数阈值来识别混合系统模式切换的根本问题，该研究提出了一种创新的切入视角——即模式切换的本质是单一动态模型的拟合失效——并以此为基础，通过将非线性自回归（NARX）模型的拟合残差作为一种统一、无阈值的内在标准，成功地实现了对高阶非线性混合系统结构与动态的精确推断。

*   **一句话总结 (大白话版)**
    要判断一个机器人啥时候切换了新任务，别费劲去看它的速度是不是突然变快了，你只要一直用同一个“行为说明书”去套它的动作，当说明书突然对不上号的那一刻，就说明它换任务了。