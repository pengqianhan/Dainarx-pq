# 基于拟合能力的聚类算法详解

## 概述

这是 DAINARX 方法的核心创新：**不是比较参数向量的距离，而是测试"能否用同一组 NARX 参数拟合不同数据段"**。聚类算法实现了论文中的**可合并性（Mergeable）**和**最小可合并性（Minimally Mergeable）**概念。

---

## 1. 理论基础

### 1.1 论文定义（Definition 11 & 13）

**可合并性（Mergeable Traces）**：
一组离散时间轨迹段集合 $S$ 对于模板 NARX 模型 $N$ 是可合并的，当且仅当存在 $N \in \langle N \rangle$ 使得 $N \models S$（即该 NARX 模型能拟合所有这些段）。

**最小可合并性（Minimally Mergeable Traces）**：
集合 $S$ 是最小可合并的，如果拟合 $S$ 所需的最小 NARX 阶数 $k_S$ 等于拟合任意单个段 $\xi \in S$ 所需的阶数 $k_{\{\xi\}}$。

**数学表达**：
$$k_S = \min\{k \in \mathbb{N} | k \text{ 是 } N \in \langle N \rangle \text{ 的阶数且 } N \models S\}$$

**判断条件**：
$$S \text{ 是最小可合并的 } \iff k_S = k_{\{\xi\}} \text{ 对所有 } \xi \in S \text{ 成立}$$

---

## 2. 算法整体结构

### 2.1 主函数签名

```python
def clustering(data: list[Slice], self_loop=False):
    """
    基于 NARX 拟合能力的聚类算法
    
    参数：
        data: Slice 对象列表，每个 Slice 代表一个轨迹分段
        self_loop: 是否允许自环（同一模式到自身的转换）
    
    核心思想：
        - 能被同一组 NARX 模型拟合的数据段 → 同一模式
        - 维护 mode_dict[mode_id] = [代表性数据段列表]（最多3个）
        - 使用延迟列表处理歧义情况
    """
```

### 2.2 算法流程图

```
输入: [Slice_0, Slice_1, Slice_2, ..., Slice_N]
      
初始化: mode_dict = {}, tot_mode = 1, delay_list = []

┌─────────────────────────────────────────────────┐
│  第一阶段：顺序聚类                              │
└─────────────────────────────────────────────────┘
对于每个 Slice_i:
  ├─ 是否 valid? ──否──> 跳过
  │        是
  ├─ 是否 isFront? ──是──> last_mode = None
  │        否
  ├─ 尝试与现有模式匹配
  │   ├─ test_set(mode_1) ──通过──> 分配到 mode_1
  │   ├─ test_set(mode_2) ──通过──> 分配到 mode_2
  │   └─ ...
  │
  ├─ 匹配结果:
  │   ├─ 0个匹配 ──> 创建新模式 (tot_mode++)
  │   ├─ 1个匹配 ──> 分配到该模式，更新 mode_dict
  │   └─ 多个匹配 ──> 加入 delay_list（歧义）
  │
  └─ 更新 last_mode (if not self_loop)

┌─────────────────────────────────────────────────┐
│  第二阶段：处理延迟列表                          │
└─────────────────────────────────────────────────┘
对于 delay_list 中的每个 Slice:
  ├─ 重新尝试匹配（排除前一段的模式）
  ├─ 匹配成功 ──> 分配到该模式
  └─ 仍失败 ──> 创建新模式

输出: 每个 Slice 都被分配了 mode ID
```

---

## 3. 第一阶段：顺序聚类

### 3.1 初始化与预处理

```python
for i in range(len(data)):
    data[i].idx = i           # 为每个 Slice 设置索引
    if not data[i].valid:     # 跳过无效分段（拟合误差过大）
        continue
    if data[i].isFront:       # 如果是轨迹起始段，重置上下文
        last_mode = None
```

### 3.2 两种聚类方法

#### 方法A：基于参数距离的聚类 (`Method == 'dis'`)

```python
if Slice.Method == 'dis':
    for j in range(i):
        # 与之前所有分段比较（排除 last_mode）
        if (data[j].mode != last_mode) and (data[i] & data[j]):
            data[i].setMode(data[j].mode)
        if data[i].mode is not None:
            break
```

**`&` 运算符定义** (src/CurveSlice.py:106-115)：

```python
def __and__(self, other):
    """判断两个 Slice 是否属于同一模式（距离方法）"""
    if Slice.Method == 'dis':
        idx = 0
        for v1, v2 in zip(self.feature, other.feature):
            relative_dis, dis = Slice.get_dis(v1, v2)
            # 相对距离和绝对距离至少有一个小于阈值
            if relative_dis > Slice.RelativeErrorThreshold[idx] and \
                    dis > Slice.AbsoluteErrorThreshold[idx]:
                return False
            idx += 1
        return True
```

**距离计算公式**：

```python
@staticmethod
def get_dis(v1, v2):
    dis = np.linalg.norm(v1 - v2, ord=1)  # L1 绝对距离
    d1 = np.linalg.norm(v1, ord=1)
    d2 = np.linalg.norm(v2, ord=1)
    d_min = min(d1, d2)
    relative_dis = dis / max(d_min, 1e-6)  # 相对距离
    return relative_dis, dis
```

**数学表达**：
$$\text{dis}(v_1, v_2) = \|v_1 - v_2\|_1 = \sum_{i} |v_{1,i} - v_{2,i}|$$

$$\text{relative\_dis}(v_1, v_2) = \frac{\|v_1 - v_2\|_1}{\max(\min(\|v_1\|_1, \|v_2\|_1), 10^{-6})}$$

**判断条件**：
$$v_1 \& v_2 = \text{True} \iff \forall i: (\text{relative\_dis}_i \leq \theta_{\text{rel}}[i]) \lor (\text{dis}_i \leq \theta_{\text{abs}}[i])$$

#### 方法B：基于拟合能力的聚类 (`Method == 'fit'`) - **推荐方法**

```python
else:  # 'fit' 方法
    fit_cnt = 0
    # 遍历已识别的所有模式
    for idx, val in mode_dict.items():
        # 测试1：当前数据段能否与该模式的代表段合并拟合？
        if idx != last_mode and data[i].test_set(val):
            data[i].mode = idx
            fit_cnt += 1
        
        # 测试2：处理模式定义不稳定的情况（只有2个代表段）
        elif len(val) == 2:
            # 尝试单独与第一个或第二个代表段拟合
            if data[i].test_set([val[0]]) or data[i].test_set([val[1]]):
                # 说明该模式定义不够稳定，将最后一个代表段移到延迟列表
                delay_list.append(val[-1])
                mode_dict[idx].pop()
                fit_cnt = 2  # 标记为歧义情况
```

### 3.3 核心测试函数：test_set

```python
def test_set(self, other_list):
    """测试当前数据段能否与 other_list 中的段共同拟合"""
    data, input_data, other_fit_order = [], [], None
    
    # 收集参考段的数据和拟合阶数
    for s in other_list:
        data.append(s.data)
        input_data.append(s.input_data)
        if other_fit_order is None:
            other_fit_order = copy.copy(s.fit_order)
        else:
            other_fit_order = min(other_fit_order, s.fit_order)
    
    # 尝试用单一 NARX 模型拟合当前段+所有参考段
    _, err, fit_order = self.get_feature(
        [self.data] + data,
        [self.input_data] + input_data,
        is_list=True
    )
    
    # 判断条件1：拟合阶数不增加（最小可合并性）
    order_condition = True
    for i in range(len(fit_order)):
        order_condition = order_condition and \
            fit_order[i] <= max(self.fit_order[i], other_fit_order[i])
    
    # 判断条件2：拟合误差小于校准的阈值
    return order_condition and max(err) < Slice.FitErrorThreshold
```

**数学原理**：

对于当前分段 $s_i$ 和模式 $q$ 的代表性分段集合 $S_q = \{s_{q,1}, s_{q,2}, ...\}$，执行联合 NARX 拟合：

$$\min_{\Lambda} \sum_{s \in \{s_i\} \cup S_q} \|O_s - \Lambda \cdot D_s\|_2^2$$

其中 $O_s$ 是观测值矩阵，$D_s$ 是数据矩阵。

**可合并性判断条件**：

1. **阶数条件（Order Condition）**：
   $$k_{\text{combined}} \leq \max(k_{s_i}, \min_{s \in S_q} k_s)$$
   
   合并后的拟合阶数不能超过原有阶数的最大值，对应论文中的**最小可合并性**。

2. **误差条件（Error Condition）**：
   $$\max(\text{err}) < \text{FitErrorThreshold}$$
   
   合并拟合的最大残差必须小于动态校准的阈值。

**返回值**：
$$\text{test\_set}(s_i, S_q) = (\text{order\_condition}) \land (\max(\text{err}) < \theta_{\text{fit}})$$

### 3.4 为什么这样设计有效？

如果两个数据段来自同一个动力学模式 $q$，那么它们都满足相同的 NARX 方程：
$$\vec{x}[\tau] = F_q(\vec{x}[\tau-1], ..., \vec{x}[\tau-k], \vec{u}[\tau])$$

因此用单一 NARX 模型联合拟合它们应该：
1. **不需要增加模型阶数**：同一物理规律下，时间依赖深度应该一致
2. **拟合误差保持在容忍范围内**：同一动力学方程产生的数据应该能被统一描述

这种判断方式**直接基于物理意义**，而不是参数空间的几何距离，因此对噪声和参数扰动更鲁棒。

### 3.5 分配结果处理

```python
# 情况1：恰好匹配一个模式
if fit_cnt == 1:
    if len(mode_dict[data[i].mode]) < 3:
        # 添加为代表段（每个模式最多保留3个代表段）
        mode_dict[data[i].mode].append(data[i])

# 情况2：匹配多个模式（歧义）
elif fit_cnt > 1:
    delay_list.append(data[i])
    data[i].mode = -1  # 标记为未确定
```

### 3.6 创建新模式

```python
# 情况3：不匹配任何现有模式 → 创建新模式
if data[i].mode is None:
    data[i].mode = tot_mode
    mode_dict[tot_mode] = [data[i]]  # 初始化代表段列表
    tot_mode += 1
```

### 3.7 更新上下文

```python
if not self_loop:
    last_mode = data[i].mode  # 记录当前模式，避免下一段被分到同一模式
```

**设计要点**：
- **代表段限制**：每个模式最多保留3个代表段，平衡计算效率和准确性
- **时序约束**：通过 `last_mode` 避免连续分段被分到同一模式（除非 `self_loop=True`）
- **延迟处理**：歧义情况（匹配多个模式）延迟到第二阶段处理

---

## 4. 第二阶段：处理延迟列表

**仅适用于拟合方法**：

```python
# 'dis' 方法不需要第二阶段处理，直接返回
if Slice.Method != 'fit':
    return

# 第二阶段：处理歧义数据段
for s in delay_list:
    for idx, val in mode_dict.items():
        # 利用时序信息：跳过前一段的模式（避免自环）
        if not s.isFront and data[s.idx - 1].mode == idx:
            continue
        
        # 重新测试可合并性
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

**第二阶段的作用**：
1. **利用更多上下文**：第一阶段结束后，所有模式的代表段更完整
2. **时序约束**：排除前一个分段的模式，利用时间序列的因果关系
3. **最终兜底**：仍无法分类的段创建独立模式

---

## 5. 算法示例演示

假设有6个数据段，真实模式归属为 `[A, A, B, A, B, C]`：

### 第一阶段处理过程

```
段0: mode=None → 创建 mode_1, mode_dict={1: [段0]}
     last_mode=1

段1: test_set([段0]) → 通过 (同为模式A)
     mode=1, mode_dict={1: [段0, 段1]}
     last_mode=1

段2: test_set([段0, 段1]) → 失败 (模式B ≠ A)
     mode=None → 创建 mode_2, mode_dict={1: [段0, 段1], 2: [段2]}
     last_mode=2

段3: test_set([段0, 段1]) → 通过 (同为模式A)
     mode=1, mode_dict={1: [段0, 段1, 段3]}
     last_mode=1

段4: test_set([段0, 段1, 段3]) → 失败 (模式B ≠ A)
     test_set([段2]) → 通过 (同为模式B)
     mode=2, mode_dict={1: [段0, 段1, 段3], 2: [段2, 段4]}
     last_mode=2

段5: test_set([段0, 段1, 段3]) → 失败 (模式C ≠ A)
     test_set([段2, 段4]) → 失败 (模式C ≠ B)
     mode=None → 创建 mode_3, mode_dict={1: [...], 2: [...], 3: [段5]}
     last_mode=3
```

### 最终结果

- **模式1（A）**：段0, 段1, 段3
- **模式2（B）**：段2, 段4
- **模式3（C）**：段5

---

## 6. 算法复杂度分析

### 时间复杂度

- **第一阶段**：$O(N \cdot M \cdot C)$
  - $N$：数据段总数
  - $M$：平均模式数量
  - $C$：单次 NARX 拟合复杂度 $O(w \cdot d^2)$
- **第二阶段**：$O(|D| \cdot M \cdot C)$，其中 $|D|$ 是延迟列表大小（通常 $|D| \ll N$）

### 空间复杂度

- `mode_dict`：$O(M \cdot 3)$（每个模式最多3个代表段）
- `delay_list`：$O(|D|)$

### 实际性能

- 对于典型混合系统（$N \sim 100$, $M \sim 5$, $w \sim 10$），算法在毫秒级完成
- 代表段限制（最多3个）显著降低计算开销

---

## 7. 两种方法对比

| 特性 | 距离方法 (`dis`) | 拟合方法 (`fit`) |
|------|-----------------|------------------|
| **判断依据** | 参数向量的 L1 距离 | NARX 模型的可合并性 |
| **理论基础** | 几何距离 | 物理意义（最小可合并性） |
| **计算复杂度** | 低（$O(d)$） | 中等（$O(w \cdot d^2)$） |
| **鲁棒性** | 对噪声敏感 | 对噪声鲁棒 |
| **参数扰动** | 敏感 | 不敏感 |
| **延迟处理** | 无 | 有（第二阶段） |
| **推荐场景** | 快速原型、低噪声数据 | 生产环境、高噪声数据 |

**推荐使用 `fit` 方法**，因为它：
1. 直接对应论文理论（可合并性）
2. 基于物理意义而非几何距离
3. 对噪声和参数扰动更鲁棒
4. 利用时序信息处理歧义

---

## 8. 聚类过程可视化

```
时间序列: |====模式A====|****|====模式B====|****|====模式A====|****|====模式C====|
分段结果:  [段0][段1][段2]    [段3][段4]         [段5][段6]         [段7]

聚类过程:
段0 → 创建 mode_1
段1 → test_set([段0]) ✓ → mode_1
段2 → test_set([段0,段1]) ✓ → mode_1
段3 → test_set([段0,段1,段2]) ✗ → 创建 mode_2
段4 → test_set([段0,段1,段2]) ✗, test_set([段3]) ✓ → mode_2
段5 → test_set([段0,段1,段2]) ✓ → mode_1
段6 → test_set([段0,段1,段2]) ✓ → mode_1
段7 → test_set([段0,段1,段2]) ✗, test_set([段3,段4]) ✗ → 创建 mode_3

最终聚类:
mode_1 (A): [段0, 段1, 段2, 段5, 段6]
mode_2 (B): [段3, 段4]
mode_3 (C): [段7]
```

---

## 9. 关键设计洞察

1. **贪心策略的合理性**：顺序处理数据段，利用时间序列的因果关系，避免全局优化的计算开销

2. **代表段机制**：每个模式保留最多3个代表段，既保证了模式定义的稳定性，又控制了计算复杂度

3. **延迟处理策略**：将歧义情况延迟到第二阶段，利用更完整的上下文信息做出更准确的判断

4. **时序约束**：通过 `last_mode` 和 `isFront` 利用时间序列的结构信息，提高聚类准确性

5. **物理意义优先**：拟合方法基于"能否用同一动力学方程描述"而非"参数向量是否接近"，更符合混合系统的本质

---

## 10. 与论文理论的对应关系

| 论文概念 | 代码实现 | 位置 |
|---------|---------|------|
| **可合并性（Mergeable）** | `test_set()` 返回 True | `CurveSlice.py:90-104` |
| **最小可合并性** | `order_condition` 检查 | `CurveSlice.py:113-116` |
| **模式识别** | `mode_dict` 维护 | `Clustering.py:17` |
| **贪心聚类** | 顺序遍历 `data` | `Clustering.py:20-66` |
| **歧义处理** | `delay_list` 机制 | `Clustering.py:48-60` |

---

## 总结

`clustering` 函数实现了一个**贪心的、基于 NARX 拟合的聚类算法**：

1. 顺序处理每个分段
2. 尝试与已有模式合并（通过 NARX 拟合测试）
3. 无法合并则创建新模式
4. 处理歧义情况（延迟列表）
5. 最终每个分段都被分配到一个模式，代表不同的动力学行为

该算法的核心优势在于**直接基于物理意义（动力学方程的一致性）而非几何距离进行聚类**，这使得它对噪声和参数扰动具有更强的鲁棒性，完美契合混合系统识别的本质需求。

