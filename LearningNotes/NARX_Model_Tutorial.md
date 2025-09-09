# NARX模型详解教程：以弹跳球系统为例

## 目录
1. [什么是NARX模型](#1-什么是narx模型)
2. [弹跳球：经典的混合系统](#2-弹跳球经典的混合系统)
3. [从连续时间到NARX离散模型](#3-从连续时间到narx离散模型)
4. [NARX模型的数学表示](#4-narx模型的数学表示)
5. [特征提取与线性回归](#5-特征提取与线性回归)
6. [模式切换检测](#6-模式切换检测)
7. [重置函数处理](#7-重置函数处理)
8. [完整学习流程](#8-完整学习流程)
9. [代码实现示例](#9-代码实现示例)
10. [理论优势总结](#10-理论优势总结)

---

## 1. 什么是NARX模型

NARX（Nonlinear AutoRegressive eXogenous）模型是一种强大的时间序列建模方法，特别适用于混合系统的推断。

### 1.1 基本概念

**NARX模型的核心思想**：当前时刻的系统状态可以通过历史状态和外部输入的非线性函数来预测。

**数学表达**：
```
y[t] = F(y[t-1], y[t-2], ..., y[t-k], u[t-1], u[t-2], ..., u[t-m]) + ε[t]
```

其中：
- `y[t]`：当前时刻的输出
- `y[t-i]`：历史输出值
- `u[t-j]`：外部输入值
- `F(·)`：非线性映射函数
- `ε[t]`：噪声项
- `k,m`：回归阶数

### 1.2 为什么选择NARX？

1. **无需导数**：直接使用离散时间数据，避免数值微分
2. **统一框架**：同时处理连续动力学和离散事件
3. **自适应性**：通过拟合质量自动检测模式变化
4. **计算高效**：转化为线性代数问题

---

## 2. 弹跳球：经典的混合系统

### 2.1 物理系统描述

弹跳球系统是理解混合系统的经典例子：

**状态变量**：
- `x1`：球的高度（位置）[m]
- `x2`：球的速度 [m/s]

**物理参数**：
- 重力加速度：`g = 9.8 m/s²`
- 反弹系数：`e = 0.9` （能量损失10%）

### 2.2 系统配置文件

```json
{
  "automaton": {
    "var": "x1, x2",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -9.8"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x1 <= 0",
        "reset": {
          "x1": [0],
          "x2": ["-0.9 * x2[0]"]
        }
      }
    ]
  },
  "config": {
    "dt": 0.01,
    "total_time": 10.0,
    "order": 1,
    "other_items": "x_[?]"
  }
}
```

### 2.3 混合系统特性

**连续动力学**（空中飞行）：
```
dx1/dt = x2        (位置变化率 = 速度)
dx2/dt = -9.8      (速度变化率 = 重力)
```

**离散事件**（碰撞地面）：
- **触发条件**：`x1 ≤ 0`（球触地）
- **状态重置**：
  - `x1 := 0`（位置重置）
  - `x2 := -0.9 × x2`（速度反向衰减）

---

## 3. 从连续时间到NARX离散模型

### 3.1 离散化必要性

传统的连续时间建模面临挑战：
1. **微分方程求解复杂**：需要数值积分方法
2. **导数计算误差**：数值微分引入噪声
3. **混合事件处理困难**：连续与离散的混合

### 3.2 NARX离散化过程

**采样间隔**：`dt = 0.01s`

**连续方程的离散近似**：
```
x1[k+1] ≈ x1[k] + dt × x2[k]
x2[k+1] ≈ x2[k] + dt × (-9.8)
```

即：
```
x1[k+1] ≈ x1[k] + 0.01 × x2[k]
x2[k+1] ≈ x2[k] - 0.098
```

### 3.3 NARX统一表示

**一阶NARX形式**：
```
x1[k+1] = f1(x1[k], x2[k])
x2[k+1] = f2(x1[k], x2[k])
```

**线性化近似**：
```
x1[k+1] = a11×x1[k] + a12×x2[k] + b1
x2[k+1] = a21×x1[k] + a22×x2[k] + b2
```

---

## 4. NARX模型的数学表示

### 4.1 通用NARX公式

对于k阶NARX模型：
```
x[τ] = Σ(i=1 to α) ai ○ fi(x[τ-1], ..., x[τ-k], u[τ]) 
     + Σ(i=1 to k) Bi × x[τ-i] 
     + Bk+1 × u[τ] + c
```

其中：
- `ai, c`：向量系数
- `Bi, Bk+1`：矩阵系数
- `fi`：非线性函数
- `○`：Hadamard积

### 4.2 弹跳球的NARX实例化

**状态空间**：2维 (`x1`, `x2`)
**系统阶数**：1阶
**非线性项**：无（线性系统）

**具体形式**：
```
[x1[k+1]]   [a11 a12] [x1[k]]   [b1]
[x2[k+1]] = [a21 a22] [x2[k]] + [b2]
```

### 4.3 特征向量构建

对于每个时间步，特征向量为：
```
φ[k] = [x1[k], x2[k], 1]  # [状态变量 + 偏置项]
```

目标向量为：
```
t1[k] = x1[k+1]  # 下一时刻的位置
t2[k] = x2[k+1]  # 下一时刻的速度
```

---

## 5. 特征提取与线性回归

### 5.1 训练数据构建

假设采集到的时间序列数据：
```
时刻0: x1=1.20, x2=-2.000
时刻1: x1=1.18, x2=-2.098  
时刻2: x1=1.159, x2=-2.196
时刻3: x1=1.137, x2=-2.294
...
```

### 5.2 特征矩阵构建

```python
# 特征矩阵 Φ (每行一个时间步的特征)
Φ = [[1.20,  -2.000, 1],    # 时刻0
     [1.18,  -2.098, 1],    # 时刻1  
     [1.159, -2.196, 1],    # 时刻2
     [1.137, -2.294, 1],    # 时刻3
     ...]

# 目标矩阵
T_x1 = [1.18, 1.159, 1.137, ...]   # x1的下一时刻值
T_x2 = [-2.098, -2.196, -2.294, ...] # x2的下一时刻值
```

### 5.3 线性最小二乘求解

**求解x1的系数**：
```python
θ_x1 = (Φ^T Φ)^(-1) Φ^T T_x1
```

**求解x2的系数**：
```python
θ_x2 = (Φ^T Φ)^(-1) Φ^T T_x2
```

### 5.4 预期结果

理想情况下应该得到：
```python
θ_x1 ≈ [1.0, 0.01, 0]      # x1[k+1] = x1[k] + 0.01×x2[k]
θ_x2 ≈ [0.0, 1.0, -0.098]  # x2[k+1] = x2[k] - 0.098
```

---

## 6. 模式切换检测

### 6.1 传统方法的局限

**基于导数的方法**：
- 计算速度导数：`dx2/dt`
- 检测导数突变（碰撞时从-9.8突变为正值）
- **问题**：需要设置阈值，对噪声敏感

### 6.2 NARX的无阈值检测

**基本思想**：通过拟合质量检测模式切换

```python
def detect_mode_switch(current_state, narx_model):
    # 使用当前NARX模型预测下一状态
    predicted_x1 = narx_model.predict_x1(current_state)
    predicted_x2 = narx_model.predict_x2(current_state)
    
    # 计算实际观测值
    actual_x1, actual_x2 = get_next_observation()
    
    # 计算预测误差
    error_x1 = abs(predicted_x1 - actual_x1)
    error_x2 = abs(predicted_x2 - actual_x2)
    
    # 误差突然增大表示模式切换
    if error_x1 > adaptive_threshold or error_x2 > adaptive_threshold:
        return True  # 检测到模式切换
    return False
```

### 6.3 自适应阈值

NARX方法的优势是可以使用自适应阈值：
```python
# 基于历史拟合误差的自适应阈值
adaptive_threshold = mean(historical_errors) + 3 × std(historical_errors)
```

---

## 7. 重置函数处理

### 7.1 重置事件建模

碰撞时的状态跳跃：
```
x1^- → x1^+ = 0
x2^- → x2^+ = -0.9 × x2^-
```

### 7.2 重置的NARX表示

将重置建模为特殊的"瞬时模式"：

```python
def reset_narx_model(state_before_collision):
    x1_before, x2_before = state_before_collision
    
    # 重置函数的NARX形式
    x1_after = 0                      # 位置重置为0
    x2_after = -0.9 * x2_before       # 速度反向衰减
    
    return [x1_after, x2_after]
```

### 7.3 重置学习过程

```python
# 收集碰撞前后的状态对
collision_data = []
for collision_event in collision_events:
    state_before = collision_event.state_before
    state_after = collision_event.state_after
    collision_data.append((state_before, state_after))

# 学习重置函数
# 对于x1: x1_after = c1 (常数0)
# 对于x2: x2_after = c2 × x2_before (系数-0.9)
reset_coeffs = learn_reset_function(collision_data)
```

---

## 8. 完整学习流程

### 8.1 数据预处理

```python
def preprocess_trajectory(raw_trajectory):
    """
    预处理轨迹数据，识别连续段和碰撞点
    """
    continuous_segments = []
    collision_points = []
    
    for i in range(len(raw_trajectory) - 1):
        current_state = raw_trajectory[i]
        next_state = raw_trajectory[i + 1]
        
        # 检测碰撞（高度从正值变为0，速度方向改变）
        if current_state[0] > 0 and next_state[0] == 0:
            collision_points.append(i)
        else:
            continuous_segments.append(i)
    
    return continuous_segments, collision_points
```

### 8.2 模式识别

```python
def identify_modes(trajectory_segments):
    """
    将轨迹段聚类为不同的模式
    """
    # 对于弹跳球，所有连续段都属于同一"飞行模式"
    flight_mode_segments = []
    
    for segment in trajectory_segments:
        # 使用NARX模型拟合每个段
        narx_fit = fit_narx_model(segment)
        flight_mode_segments.append(narx_fit)
    
    # 验证所有段是否可以用同一模型拟合
    unified_model = fit_unified_narx_model(flight_mode_segments)
    
    return unified_model
```

### 8.3 综合模型学习

```python
def learn_bouncing_ball_model(trajectory_data):
    """
    学习完整的弹跳球混合系统模型
    """
    # 步骤1: 轨迹分割
    continuous_segments, collision_points = preprocess_trajectory(trajectory_data)
    
    # 步骤2: 学习飞行模式的NARX模型
    flight_model = learn_flight_narx(continuous_segments)
    
    # 步骤3: 学习重置函数
    reset_function = learn_reset_function(collision_points)
    
    # 步骤4: 学习守卫条件
    guard_condition = learn_guard_condition(collision_points)  # x1 <= 0
    
    # 步骤5: 构建完整的混合自动机
    hybrid_automaton = build_hybrid_automaton(
        flight_model, reset_function, guard_condition
    )
    
    return hybrid_automaton
```

---

## 9. 代码实现示例

### 9.1 NARX特征提取器

```python
import numpy as np

class BouncingBallNARX:
    def __init__(self, order=1):
        self.order = order
        self.flight_coeffs = None
        self.reset_coeffs = None
    
    def extract_features(self, trajectory):
        """提取NARX特征"""
        features = []
        targets_x1 = []
        targets_x2 = []
        
        for i in range(self.order, len(trajectory)):
            # 构建特征向量 [x1[k], x2[k], 1]
            feature_vector = []
            for j in range(self.order):
                feature_vector.extend(trajectory[i-j-1])  # x1[k-j], x2[k-j]
            feature_vector.append(1)  # 偏置项
            
            features.append(feature_vector)
            targets_x1.append(trajectory[i][0])  # x1[k+1]
            targets_x2.append(trajectory[i][1])  # x2[k+1]
        
        return np.array(features), np.array(targets_x1), np.array(targets_x2)
    
    def fit_flight_model(self, trajectory_segments):
        """拟合飞行模式的NARX模型"""
        all_features = []
        all_targets_x1 = []
        all_targets_x2 = []
        
        # 合并所有连续段的数据
        for segment in trajectory_segments:
            features, targets_x1, targets_x2 = self.extract_features(segment)
            all_features.extend(features)
            all_targets_x1.extend(targets_x1)
            all_targets_x2.extend(targets_x2)
        
        X = np.array(all_features)
        y1 = np.array(all_targets_x1)
        y2 = np.array(all_targets_x2)
        
        # 线性最小二乘求解
        self.flight_coeffs_x1 = np.linalg.lstsq(X, y1, rcond=None)[0]
        self.flight_coeffs_x2 = np.linalg.lstsq(X, y2, rcond=None)[0]
        
        return self.flight_coeffs_x1, self.flight_coeffs_x2
    
    def predict_next_state(self, current_state):
        """预测下一时刻状态"""
        if self.flight_coeffs_x1 is None:
            raise ValueError("Model not trained yet")
        
        # 构建特征向量
        feature_vector = list(current_state) + [1]
        feature_vector = np.array(feature_vector)
        
        # 预测
        x1_next = np.dot(feature_vector, self.flight_coeffs_x1)
        x2_next = np.dot(feature_vector, self.flight_coeffs_x2)
        
        return [x1_next, x2_next]
    
    def apply_reset(self, state_before_collision):
        """应用重置函数"""
        x1_before, x2_before = state_before_collision
        
        # 弹跳球的重置规则
        x1_after = 0              # 位置重置为地面
        x2_after = -0.9 * x2_before  # 速度反向衰减
        
        return [x1_after, x2_after]
    
    def simulate_trajectory(self, initial_state, time_steps):
        """模拟轨迹"""
        trajectory = [initial_state]
        current_state = initial_state.copy()
        
        for _ in range(time_steps):
            # 预测下一状态
            next_state = self.predict_next_state(current_state)
            
            # 检查是否触发守卫条件（撞地）
            if next_state[0] <= 0:
                # 应用重置
                next_state = self.apply_reset(current_state)
            
            trajectory.append(next_state)
            current_state = next_state
        
        return np.array(trajectory)
```

### 9.2 使用示例

```python
# 创建NARX模型
narx_model = BouncingBallNARX(order=1)

# 模拟训练数据
def generate_training_data():
    """生成弹跳球训练数据"""
    # 初始条件
    x1_0, x2_0 = 1.0, 0.0  # 从1米高度静止释放
    dt = 0.01
    
    trajectory = []
    x1, x2 = x1_0, x2_0
    
    for i in range(1000):  # 10秒数据
        trajectory.append([x1, x2])
        
        # 数值积分（欧拉方法）
        x1_next = x1 + dt * x2
        x2_next = x2 + dt * (-9.8)
        
        # 检查碰撞
        if x1_next <= 0:
            x1_next = 0
            x2_next = -0.9 * x2  # 反弹
        
        x1, x2 = x1_next, x2_next
    
    return np.array(trajectory)

# 生成训练数据
training_data = generate_training_data()

# 分割连续段（简化处理）
continuous_segments = [training_data]  # 实际应用中需要更智能的分割

# 训练模型
coeffs_x1, coeffs_x2 = narx_model.fit_flight_model(continuous_segments)

print(f"x1系数: {coeffs_x1}")  # 期望: [1.0, 0.01, 0]
print(f"x2系数: {coeffs_x2}")  # 期望: [0.0, 1.0, -0.098]

# 模拟预测
initial_state = [1.2, -1.5]
predicted_trajectory = narx_model.simulate_trajectory(initial_state, 500)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(predicted_trajectory[:, 0], label='Position (x1)')
plt.xlabel('Time steps')
plt.ylabel('Height [m]')
plt.title('Ball Position')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(predicted_trajectory[:, 1], label='Velocity (x2)', color='red')
plt.xlabel('Time steps')
plt.ylabel('Velocity [m/s]')
plt.title('Ball Velocity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 10. 理论优势总结

### 10.1 与传统方法对比

| 方面 | 传统微分方程方法 | NARX方法 |
|------|------------------|----------|
| **数据要求** | 需要导数信息 | 仅需离散观测 |
| **噪声鲁棒性** | 对导数噪声敏感 | 对观测噪声鲁棒 |
| **模式检测** | 基于导数突变+阈值 | 基于拟合质量+无阈值 |
| **计算复杂度** | 需要数值积分 | 线性代数运算 |
| **参数调节** | 需要手动设置多个阈值 | 参数自适应学习 |
| **扩展性** | 复杂系统难以建模 | 统一框架易扩展 |

### 10.2 核心优势

1. **无导数特性**
   - 避免数值微分的误差累积
   - 直接处理采样数据
   - 适合实际测量系统

2. **统一表示框架**
   - 连续动力学和离散事件统一处理
   - 线性和非线性动力学统一建模
   - 多模态系统统一学习

3. **自动化推断**
   - 模式切换自动检测
   - 参数自动优化
   - 无需领域专家知识

4. **计算高效性**
   - 转化为成熟的线性代数问题
   - 可利用现有优化算法
   - 适合大规模数据处理

### 10.3 适用场景

**NARX方法特别适用于**：
- 混合动力系统（如机器人控制）
- 过程工业系统（如化工反应器）
- 生物医学系统（如心律监测）
- 经济金融系统（如市场切换模型）
- 故障检测系统（如设备状态监控）

### 10.4 局限性与改进

**当前局限**：
1. 对非线性项的选择依赖经验
2. 高维系统的计算复杂度
3. 长期预测的累积误差

**改进方向**：
1. 结合机器学习自动选择非线性项
2. 采用正则化技术防止过拟合
3. 使用集成学习提高预测精度

---

## 结论

通过弹跳球这个经典例子，我们深入理解了NARX模型的核心思想和实现方法。NARX将复杂的混合系统推断问题转化为可解的线性回归问题，为混合系统的自动建模提供了强有力的工具。

这种方法的成功应用不仅体现在弹跳球这样的简单系统中，更重要的是为处理复杂的实际工程系统奠定了理论基础。随着数据驱动方法的发展，NARX模型必将在更多领域发挥重要作用。

---

## 参考资料

1. Yu et al. (2025). "Derivative-Agnostic Inference of Nonlinear Hybrid Systems". ACM Trans. Embedd. Comput. Syst.
2. Billings, S.A. (2013). "Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains". John Wiley & Sons.
3. Lygeros, J. et al. (2003). "Dynamical properties of hybrid automata". IEEE Trans. Autom. Control.
4. Dainarx项目: https://github.com/FICTION-ZJU/Dainarx

---

*本教程完整展示了NARX模型从理论到实践的全过程，为读者提供了深入理解和实际应用的基础。*