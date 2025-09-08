# FeatureExtractor Tutorial: 从理论到实现

## 概述

FeatureExtractor类是Dainarx框架的核心组件，负责将论文中的NARX（Nonlinear AutoRegressive eXogenous）模型理论转化为可计算的线性代数问题。本教程详细解释了FeatureExtractor类与论文理论的对应关系。

## 1. 理论基础

### 1.1 NARX模型的数学表示

根据论文第3.2节，NARX模型的基本形式为：

**连续时间形式（公式3）：**
```
x[τ] = Fq(x[τ-1], x[τ-2], ..., x[τ-k], u[τ])
```

**扩展形式（公式4）：**
```
x[τ] = Σ(ai ○ fi(x[τ-1], ..., x[τ-k], u[τ])) + Σ(Bi·x[τ-i]) + Bk+1·u[τ] + c
```

其中：
- `ai, c` 为向量系数
- `Bi, Bk+1` 为矩阵系数
- `fi` 为非线性函数
- `○` 表示Hadamard积（逐元素相乘）

### 1.2 线性最小二乘优化

论文第7页提出的优化问题（公式5）：
```
minimize ||Oi,: - Λi,: · Di||² for i = 1, 2, ..., n
```

其中：
- `O` 为观测值矩阵
- `Di` 为数据矩阵
- `Λi,:` 为待求解的系数向量

## 2. FeatureExtractor类结构

### 2.1 核心方法概览

| 方法名 | 功能 | 对应论文内容 |
|--------|------|-------------|
| `__init__` | 初始化参数和解析模板 | 模板NARX模型定义 |
| `get_items` | 构建特征向量 | 数据矩阵Di的行构建 |
| `work_normal` | 标准NARX拟合 | 线性最小二乘求解 |
| `work_minus` | 最小阶数拟合 | Minimally mergeable概念 |
| `append_data` | 构建训练矩阵 | 观测值矩阵O构建 |
| `analyticalExpression` | 解析非线性表达式 | 模板中非线性项解析 |

### 2.2 类初始化

```python
def __init__(self, var_num: int, input_num: int, order: int, dt: float,
             need_bias: bool = False, minus: bool = False, other_items: str = ''):
```

**参数说明：**
- `var_num`: 状态变量数量（n）
- `input_num`: 输入变量数量（m）
- `order`: NARX阶数（k）
- `dt`: 采样时间间隔
- `need_bias`: 是否需要偏置项
- `minus`: 是否使用最小阶数方法
- `other_items`: 非线性项定义字符串

## 3. 特征提取核心：get_items方法

### 3.1 方法实现分析

```python
def get_items(self, data, input_data, idx, max_order=None):
    res = []
    if max_order is None:
        max_order = self.order
    max_order = min(self.order, max_order)
    
    # 1. 历史状态项：对应公式(4)中的Bi·x[τ-i]项
    for i in range(len(data[idx]) - self.order, len(data[idx]) - self.order + max_order):
        res.append(data[idx][i])
    for i in range(len(data[idx]) - self.order + max_order, len(data[idx])):
        res.append(0.)
    
    # 2. 非线性项：对应公式(4)中的ai ○ fi(...)项  
    for (fun, order) in zip(self.fun_list[idx], self.fun_order[idx]):
        if order > max_order:
            res.append(0.)
        else:
            res.append(fun(data))
    
    # 3. 输入项：对应公式(4)中的Bk+1·u[τ]项
    for input_idx in range(len(input_data)):
        res.append(input_data[input_idx][0])
    
    # 4. 偏置项：对应公式(4)中的常数项c
    if self.need_bias:
        res.append(1.)
    return res
```

### 3.2 特征向量构成

对于k阶NARX模型，特征向量包含：

1. **历史状态项**: `[x[τ-k], x[τ-k+1], ..., x[τ-1]]`
2. **非线性项**: `[f1(x[τ-1], ...), f2(x[τ-1], ...), ...]`
3. **输入项**: `[u1[τ], u2[τ], ...]`
4. **偏置项**: `[1]` (可选)

## 4. 线性最小二乘求解

### 4.1 标准拟合：work_normal方法

```python
def work_normal(self, data, input_data, is_list: bool):
    res = []
    err = []
    var_num = len(data) if not is_list else len(data[0])
    matrix_list = [[] for _ in range(var_num)]
    b_list = [[] for _ in range(var_num)]
    
    # 构建训练矩阵
    if is_list:
        for block, block_input in zip(data, input_data):
            self.append_data(matrix_list, b_list, block, block_input)
    else:
        self.append_data(matrix_list, b_list, data, input_data)
    
    # 对每个变量求解线性最小二乘
    for a, b in zip(matrix_list, b_list):
        x = np.linalg.lstsq(a, b, rcond=None)[0]  # 求解Ax=b
        res.append(x)
        err.append(max(np.abs((a @ x) - b)))  # 计算拟合误差
    
    return res, err, [self.order for _ in range(self.var_num)]
```

**实现要点：**
- 对每个状态变量独立求解
- 使用NumPy的`lstsq`函数求解线性最小二乘
- 计算拟合误差用于质量评估

### 4.2 最小阶数拟合：work_minus方法

```python
def work_minus(self, data, input_data, is_list: bool):
    res = []
    err = []
    max_order = []
    for idx in range(self.var_num):
        now_order = 1
        while True:
            # 构建当前阶数的训练数据
            a, b = [], []
            if is_list:
                for block, block_input in zip(data, input_data):
                    self.append_data_only(a, b, block, block_input, idx, now_order)
            else:
                self.append_data_only(a, b, data, input_data, idx, now_order)
            
            # 求解并计算误差
            x = np.linalg.lstsq(a, b, rcond=None)[0]
            a, b = np.array(a), np.array(b)
            now_err = max(np.abs((a @ x) - b))
            
            # 判断是否找到最小阶数
            if now_order == self.order or now_err < 1e-8:
                res.append(x)
                max_order.append(now_order)
                err.append(now_err)
                break
            now_order += 1
    return res, err, max_order
```

**关键特性：**
- 实现论文中的**minimally mergeable**概念
- 自动寻找能够拟合数据的最小阶数
- 避免过拟合问题

## 5. 数据矩阵构建：append_data方法

### 5.1 滑动窗口数据构建

```python
def append_data(self, matrix_list, b_list, data: np.array, input_data):
    data = np.array(data)
    input_data = np.array(input_data)
    
    for i in range(len(data[0]) - self.order):
        # 构建时间窗口
        if i == 0:
            this_line = data[:, (self.order - 1)::-1]  # 倒序取历史数据
            this_line_input = input_data[:, self.order::-1]
        else:
            this_line = data[:, (self.order + i - 1):(i - 1):-1]
            this_line_input = input_data[:, (self.order + i):(i - 1):-1]
        
        # 为每个变量构建特征向量和目标值
        for idx in range(len(this_line)):
            matrix_list[idx].append(self.get_items(this_line, this_line_input, idx))
            b_list[idx].append(data[idx][i + self.order])
```

**实现原理：**
- 使用滑动窗口技术构建训练样本
- 每个时间步生成一个训练样本
- 目标值为当前时刻的状态值

## 6. 非线性表达式解析

### 6.1 表达式解析流程

```python
@staticmethod
def analyticalExpression(expr_list: str, var_num, order):
    res = [[] for _ in range(var_num)]
    res_order = [[] for _ in range(var_num)]
    expr_list = expr_list.split(';')
    
    for idx in range(var_num):
        expr = FeatureExtractor.extractValidExpression(expr_list, idx)
        expr = FeatureExtractor.unfoldDigit(expr, order)
        expr = FeatureExtractor.unfoldItem(expr, idx, var_num)
        for s in expr:
            res[idx].append(eval('lambda x: ' + s))
            res_order[idx].append(FeatureExtractor.findMaxorder(s) + 1)
    
    return res, res_order
```

### 6.2 表达式展开示例

以`"x[?] ** 3"`为例（order=2）：

1. **unfoldDigit**: `"x[?] ** 3"` → `["x[0] ** 3", "x[1] ** 3"]`
2. **转换为函数**: `[lambda x: x[0]**3, lambda x: x[1]**3]`
3. **对应**: `x³[τ-2], x³[τ-1]`

## 7. 实战案例：Duffing振荡器

### 7.1 系统定义（duffing.json）

```json
{
  "automaton": {
    "mode": [
      {
        "id": 1,
        "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3"
      },
      {
        "id": 2,
        "eq": "x[2] = u - 0.2 * x[1] + x[0] - 0.5 * x[0] ** 3"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "abs(x) <= 0.8",
        "reset": {"x": ["", "x[1] * 0.95"]}
      }
    ]
  },
  "config": {
    "order": 2,
    "other_items": "x[?] ** 3"
  }
}
```

### 7.2 特征向量构建

对于Duffing系统，FeatureExtractor构建的特征向量为：
```
[x[τ-2], x[τ-1], x³[τ-2], x³[τ-1], u[τ], 1]
```

### 7.3 论文Figure 2对应关系

**Target System H (论文Figure 2a)**：
- q1 (high damping): `x(2) = u - 0.5x(1) + x - 1.5x³`
- q2 (low damping): `x(2) = u - 0.2x(1) + x - 0.5x³`

**推断结果 (论文Figure 2c)**：
- q1: `x[τ] = 2x[τ-1] - x[τ-2] - 1.5×10⁻⁶x³[τ-1] + 10⁻⁷u[τ]`
- q2: `x[τ] = 2x[τ-1] - x[τ-2] - 5×10⁻⁷x³[τ-1] + 10⁻⁷u[τ]`

这些系数由FeatureExtractor通过线性最小二乘求解得到。

## 8. 理论优势

### 8.1 无导数方法（Derivative-Agnostic）

- **传统方法**：依赖导数计算检测模式切换
- **FeatureExtractor**：直接使用离散时间序列，避免数值微分的误差

### 8.2 无阈值推断（Threshold-Free）

- **传统方法**：需要用户指定阈值参数
- **FeatureExtractor**：通过拟合质量自动判断，无需人工调参

### 8.3 统一表示

- 使用NARX模型统一处理线性和非线性动力学
- 支持高阶系统和外部输入
- 可处理多项式、有理函数、三角函数等复杂动力学

## 9. 使用指南

### 9.1 基本使用

```python
# 初始化FeatureExtractor
extractor = FeatureExtractor(
    var_num=1,      # 状态变量数
    input_num=1,    # 输入变量数
    order=2,        # 系统阶数
    dt=0.001,       # 采样间隔
    other_items="x[?] ** 3"  # 非线性项
)

# 拟合数据
coefficients, errors, orders = extractor(data, input_data)
```

### 9.2 配置参数

- **order**: 根据系统动力学阶数设置
- **other_items**: 根据系统非线性特性定义
- **minus**: 对复杂系统建议设为True以寻找最小阶数

### 9.3 质量评估

- **拟合误差**: 通过返回的errors评估拟合质量
- **系数合理性**: 检查系数量级和符号的物理意义
- **预测性能**: 在测试数据上验证模型预测能力

## 10. 总结

FeatureExtractor类成功地将论文中的NARX理论转化为可实现的算法：

1. **理论到实现**：将数学公式转化为矩阵运算
2. **自动化处理**：无需人工设置复杂参数
3. **通用性强**：支持多种类型的非线性动力学
4. **计算高效**：利用线性代数的成熟算法

这使得Dainarx能够处理传统方法无法解决的复杂混合系统推断问题，为混合系统建模提供了强大的工具。

## 参考文献

- Yu et al. (2025). Derivative-Agnostic Inference of Nonlinear Hybrid Systems. ACM Trans. Embedd. Comput. Syst.
- Dainarx源码: https://github.com/FICTION-ZJU/Dainarx