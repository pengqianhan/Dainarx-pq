# 最小二乘法在 NARX 模型中的应用示例

本目录包含了详细的示例代码，演示 `DEConfig.py:136` 行中 `np.linalg.lstsq` 的工作原理。

## 📁 文件说明

### 1. `lstsq_simple_demo.py` - 基础示例
**运行方式：** `python examples/lstsq_simple_demo.py`

**内容：**
- 示例 1：一维 AR(2) 模型拟合
- 示例 2：二维非线性系统（模拟 Duffing 振子）
- 示例 3：完整模拟 `DEConfig.py` 的 `work_normal()` 函数

**输出：** 控制台输出详细的拟合结果和参数对比

### 2. `lstsq_visual_demo.py` - 可视化示例
**运行方式：** `python examples/lstsq_visual_demo.py`

**生成文件：**
- `result/lstsq_visual_demo.png` - 9宫格完整可视化
- `result/lstsq_workflow.png` - 4步工作流程图

**可视化内容：**
1. 生成的时间序列
2. 输入信号
3. 特征矩阵 A 的热力图
4. 目标向量 b
5. 特征-目标关系散点图
6. 参数对比柱状图
7. 真实值 vs 预测值
8. 拟合误差曲线
9. 误差分布直方图

---

## 🔍 核心原理解释

### 什么是 `np.linalg.lstsq`？

`np.linalg.lstsq(A, b, rcond=None)[0]` 求解最小二乘问题：

```
找到最优参数 x，使得 ||Ax - b||² 最小
```

这是一个标准的线性回归问题，适用于样本数 > 特征数的过定方程组。

### 在 NARX 模型中的应用

#### 1. **构造回归矩阵 A**

每一行是一个时间点的特征向量：

```python
for t in range(order, T):
    feature_vector = [
        x[t-1],      # 滞后项 1
        x[t-2],      # 滞后项 2
        u[t],        # 输入项
        1.0          # 偏置项
    ]
    A.append(feature_vector)
```

**示例（order=2）：**
```
时刻 t=2: [x[1], x[0], u[2], 1]
时刻 t=3: [x[2], x[1], u[3], 1]
时刻 t=4: [x[3], x[2], u[4], 1]
...
```

**矩阵形式：**
```
     x[t-1]  x[t-2]   u[t]  bias
A = [1.200   1.000   0.065  1.0]  <- t=2
    [1.307   1.200   0.152  1.0]  <- t=3
    [1.440   1.307  -0.023  1.0]  <- t=4
    [  ...     ...     ...  ...]
```

#### 2. **构造目标向量 b**

每个元素是对应时刻的状态值：

```python
b = [x[2], x[3], x[4], ...]
```

#### 3. **求解最优参数**

```python
# 这就是 DEConfig.py:136 的核心操作！
params = np.linalg.lstsq(A, b, rcond=None)[0]
```

**返回值：**
```python
params = [a1, a2, b_input, bias]
```

#### 4. **得到 NARX 模型**

```
x[t] = a1·x[t-1] + a2·x[t-2] + b_input·u[t] + bias
```

---

## 📊 示例输出解读

### 示例 1 输出：

```
True parameters:      [0.8, 0.1, 0.5, 0.2]
Estimated parameters: [0.8846, 0.0190, 0.4715, 0.1884]
Max fitting error: 0.016621
```

**解读：**
- 估计的参数接近真实值（考虑到噪声）
- 最大拟合误差很小（0.017），说明模型拟合良好
- 这个误差会在 `DEConfig.py:138` 中用于判断模式切换

---

## 🔗 与 DEConfig.py 的对应关系

### 函数调用链：

```
work_normal()                    # DEConfig.py:124-139
    ↓
append_data()                    # DEConfig.py:147-159
    ↓
get_items()                      # DEConfig.py:80-98
    ↓
np.linalg.lstsq(A, b)[0]        # DEConfig.py:136 ← 核心！
```

### 代码对应：

| 示例代码 | DEConfig.py | 功能 |
|---------|-------------|------|
| `simulate_append_data()` | `append_data()` (147-159) | 构造矩阵 A 和向量 b |
| `feature_vector` | `get_items()` (80-98) | 构造单个特征向量 |
| `np.linalg.lstsq()` | 第 136 行 | 最小二乘求解 |
| `max(abs(A@x - b))` | 第 138 行 | 计算拟合误差 |

---

## 🎯 关键要点

### 1. **时间序列 → 回归问题**
将动态系统识别转化为标准的线性回归问题。

### 2. **滑动窗口**
通过滑动窗口提取历史数据，构造训练样本。

### 3. **支持非线性项**
通过 `get_items()` 中的 `fun_list`，可以添加任意非线性项（如 x², sin(x)）。

### 4. **拟合误差的意义**
- **小误差**：数据符合当前模型，属于同一模式
- **大误差**：模型失效，可能发生模式切换

这是论文核心创新：**用模型拟合能力判断模式切换，而非导数突变**。

---

## 🚀 快速开始

### 运行所有示例：

```bash
# 基础示例（控制台输出）
python examples/lstsq_simple_demo.py

# 可视化示例（生成图表）
python examples/lstsq_visual_demo.py
```

### 查看生成的图表：

```bash
# 打开可视化结果
start result/lstsq_visual_demo.png
start result/lstsq_workflow.png
```

---

## 📖 扩展阅读

### 相关代码文件：
- `src/DEConfig.py` - NARX 模型实现
- `src/CurveSlice.py` - 数据段管理和拟合测试
- `src/ChangePoints.py` - 基于拟合误差的切换点检测
- `src/Clustering.py` - 基于拟合能力的聚类

### 相关文档：
- `LearningNotes/paper_summary.md` - 论文方法详解
- `.cursor/rules/paper-method.mdc` - 方法论总结

---

## 💡 常见问题

### Q1: 为什么要用最小二乘法？
**A:** 最小二乘法是求解过定方程组的标准方法，能找到使拟合误差最小的参数。

### Q2: 为什么矩阵 A 的行数 > 列数？
**A:** 
- 行数 = 时间步数 - 阶数（样本数）
- 列数 = 特征数（滞后项 + 输入项 + 偏置项）
- 通常样本数远大于特征数，形成过定系统

### Q3: 拟合误差多大算"好"？
**A:** 
- 取决于数据尺度，使用 `get_eps()` 自动计算阈值
- 公式：`eps = 1e-6 * dt * max(data)`
- 如果 `max_error < eps`，说明拟合非常好

### Q4: 如何添加自定义非线性项？
**A:** 在配置文件中设置 `other_items`：
```json
{
  "other_items": "x[1]**2; x[1]*x[2]; sin(x[1])"
}
```

---

## 📝 总结

`np.linalg.lstsq` 是 NARX 模型的核心，它将时间序列预测问题转化为标准的线性回归问题。通过构造特征矩阵 A 和目标向量 b，求解最优参数，最终得到能够描述系统动态的数学模型。

**核心公式：**
```
x[t] = Σ(aᵢ·x[t-i]) + Σ(bⱼ·u[t]) + Σ(cₖ·φₖ(x,u)) + bias
```

**核心代码：**
```python
params = np.linalg.lstsq(A, b, rcond=None)[0]
```

这个简单的一行代码，支撑起了整个混合系统识别框架！

