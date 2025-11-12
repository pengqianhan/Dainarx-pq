# DEConfig.py:136 最小二乘法详解

## 问题：这段代码如何进行最小二乘？

```python
# DEConfig.py:136
x = np.linalg.lstsq(a, b, rcond=None)[0]
```

---

## 简短回答

这行代码求解**最小二乘问题**：找到最优参数向量 `x`，使得 `||Ax - b||²` 最小。

在 NARX 模型中：
- **A（矩阵）**：每行是一个时间点的特征向量 `[x[t-1], x[t-2], ..., u[t], 1]`
- **b（向量）**：每个元素是对应时刻的目标值 `x[t]`
- **x（参数）**：求解得到的模型系数 `[a1, a2, ..., b_input, bias]`

---

## 详细解释

### 1. 数据准备（`append_data` 函数）

```python
def append_data(self, matrix_list, b_list, data: np.array, input_data):
    """构造 NARX 模型的最小二乘回归矩阵"""
    data = np.array(data)
    input_data = np.array(input_data)
    
    # 滑动窗口遍历时间序列
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

**作用：** 
- 遍历时间序列，用滑动窗口提取历史数据
- 对每个时间点 t，构造特征向量并记录目标值

**示例（order=2）：**
```
时间序列: [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, ...]

t=2: 特征=[x[1], x[0], u[2], 1] → 目标=x[2]
     即：  [1.2, 1.0, 0.1, 1.0] → 1.3

t=3: 特征=[x[2], x[1], u[3], 1] → 目标=x[3]
     即：  [1.3, 1.2, 0.2, 1.0] → 1.4

t=4: 特征=[x[3], x[2], u[4], 1] → 目标=x[4]
     即：  [1.4, 1.3, 0.1, 1.0] → 1.5
...
```

**构造的矩阵：**
```
矩阵 A:                      向量 b:
[1.2  1.0  0.1  1.0]        [1.3]
[1.3  1.2  0.2  1.0]   →    [1.4]
[1.4  1.3  0.1  1.0]        [1.5]
[...]                       [...]
```

### 2. 特征构造（`get_items` 函数）

```python
def get_items(self, data, input_data, idx, max_order=None):
    res = []
    # 1. 添加线性滞后项
    for i in range(len(data[idx]) - self.order, len(data[idx])):
        res.append(data[idx][i])
    
    # 2. 添加非线性项（如 x², x₁·x₂）
    for (fun, order) in zip(self.fun_list[idx], self.fun_order[idx]):
        res.append(fun(data))
    
    # 3. 添加输入项
    for input_idx in range(len(input_data)):
        res.append(input_data[input_idx][0])
    
    # 4. 添加偏置项
    if self.need_bias:
        res.append(1.)
    
    return res
```

**作用：** 构造单个时间点的完整特征向量

**示例输出：**
```python
[x[t-1], x[t-2], x[t-1]², u[t], 1]
```

### 3. 最小二乘求解（`work_normal` 函数）

```python
def work_normal(self, data, input_data, is_list: bool):
    """使用最小二乘法拟合 NARX 模型"""
    res = []
    err = []
    var_num = len(data) if not is_list else len(data[0])
    matrix_list = [[] for _ in range(var_num)]
    b_list = [[] for _ in range(var_num)]
    
    # 步骤1: 构造回归矩阵 A 和目标向量 b
    if is_list:
        for block, block_input in zip(data, input_data):
            self.append_data(matrix_list, b_list, block, block_input)
    else:
        self.append_data(matrix_list, b_list, data, input_data)
    
    # 步骤2: 对每个变量求解 min ||Ax - b||²
    for a, b in zip(matrix_list, b_list):
        # 核心操作：最小二乘求解
        x = np.linalg.lstsq(a, b, rcond=None)[0]  # ← 这就是第136行！
        
        res.append(x)
        err.append(max(np.abs((a @ x) - b)))  # 计算最大拟合残差
    
    return res, err, [self.order for _ in range(self.var_num)]
```

**作用：**
- 调用 `append_data` 构造矩阵
- 使用 `np.linalg.lstsq` 求解最优参数
- 计算拟合误差

---

## 数学原理

### 最小二乘问题

给定：
- 矩阵 **A** (m × n)，m > n（样本数 > 特征数）
- 向量 **b** (m × 1)

求解：
```
min ||Ax - b||²
 x
```

**解析解：**
```
x = (AᵀA)⁻¹Aᵀb
```

`np.linalg.lstsq` 使用更稳定的 SVD 分解求解。

### 在 NARX 中的含义

**原始问题：** 找到 NARX 模型参数

```
x[t] = a₁·x[t-1] + a₂·x[t-2] + b·u[t] + bias
```

**转化为：** 最小二乘问题

```
[x[2]]   [x[1] x[0] u[2] 1]   [a₁]
[x[3]] ≈ [x[2] x[1] u[3] 1] × [a₂]
[x[4]]   [x[3] x[2] u[4] 1]   [b ]
[...]    [... ... ... ...]    [bias]

  b    =        A          ×    x
```

**求解：** `x = argmin ||Ax - b||²`

---

## 完整示例

```python
import numpy as np

# 1. 生成数据
T = 20
x = np.array([1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.1, ...])  # 时间序列
u = np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1, ...])  # 输入信号

# 2. 构造矩阵（order=2）
order = 2
A = []
b = []

for t in range(order, T):
    # 特征向量: [x[t-1], x[t-2], u[t], 1]
    A.append([x[t-1], x[t-2], u[t], 1.0])
    b.append(x[t])

A = np.array(A)  # shape: (18, 4)
b = np.array(b)  # shape: (18,)

# 3. 最小二乘求解（这就是 DEConfig.py:136！）
params = np.linalg.lstsq(A, b, rcond=None)[0]

# 4. 结果
print(f"参数: a1={params[0]}, a2={params[1]}, b={params[2]}, bias={params[3]}")

# 5. 计算拟合误差（DEConfig.py:138）
predictions = A @ params
max_error = np.max(np.abs(predictions - b))
print(f"最大拟合误差: {max_error}")
```

**输出示例：**
```
参数: a1=0.8846, a2=0.0190, b=0.4715, bias=0.1884
最大拟合误差: 0.016621
```

---

## 相关代码位置

### DEConfig.py

| 行号 | 函数 | 作用 |
|-----|------|------|
| 147-159 | `append_data()` | 构造回归矩阵 A 和向量 b |
| 80-98 | `get_items()` | 构造单个特征向量 |
| 124-139 | `work_normal()` | 执行最小二乘拟合 |
| **136** | **核心代码** | **`np.linalg.lstsq(a, b, rcond=None)[0]`** |
| 138 | 误差计算 | `max(np.abs((a @ x) - b))` |

### CurveSlice.py

| 行号 | 函数 | 作用 |
|-----|------|------|
| 90-104 | `test_set()` | 测试数据段能否被同一模型拟合 |
| 32-54 | `fit_threshold_one()` | 校准拟合误差阈值 |

### ChangePoints.py

| 行号 | 函数 | 作用 |
|-----|------|------|
| 16-49 | `find_change_point()` | 基于拟合误差检测切换点 |

---

## 核心思想

### 传统方法 ❌
检测导数突变 → 需要人工设定阈值 → 对噪声敏感

### 本文方法 ✅
检测模型拟合失效 → 自动计算阈值 → 鲁棒性强

**关键创新：**
```
模式切换 = 单一 NARX 模型无法继续拟合数据
```

**判断标准：**
```python
if max_error > eps:  # eps = 1e-6 * dt * max(data)
    # 发现切换点！
    change_points.append(current_position)
```

---

## 运行示例代码

```bash
# 查看详细示例
python examples/lstsq_simple_demo.py

# 生成可视化图表
python examples/lstsq_visual_demo.py

# 查看结果
start result/lstsq_visual_demo.png
start result/lstsq_workflow.png
```

---

## 总结

`np.linalg.lstsq` 将时间序列预测问题转化为标准的线性回归问题：

1. **构造矩阵 A**：每行是一个时间点的特征向量
2. **构造向量 b**：每个元素是对应的目标值
3. **求解参数 x**：使得 `||Ax - b||²` 最小
4. **得到模型**：`x[t] = a₁·x[t-1] + a₂·x[t-2] + ...`

**一句话总结：** 用历史数据预测未来值，通过最小二乘法找到最佳拟合参数！

