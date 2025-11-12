"""
NARX Model Least Squares Fitting Demo
Demonstrates how to use np.linalg.lstsq for time series modeling
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io

# Fix encoding issue on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# Example 1: Simple 1D Linear System
# ============================================================================
print("=" * 70)
print("Example 1: Simple 1D AR(2) Model")
print("=" * 70)

# True system: x[t] = 0.8*x[t-1] + 0.1*x[t-2] + 0.5*u[t] + 0.2
true_a1 = 0.8
true_a2 = 0.1
true_b = 0.5
true_bias = 0.2

# Generate training data
np.random.seed(42)
T = 20  # Number of time steps
x = np.zeros(T)
u = np.random.randn(T) * 0.1  # Input signal

# Initial conditions
x[0] = 1.0
x[1] = 1.2

# Generate time series
for t in range(2, T):
    x[t] = true_a1 * x[t-1] + true_a2 * x[t-2] + true_b * u[t] + true_bias
    x[t] += np.random.randn() * 0.01  # Add small noise

print(f"\nTrue parameters: a1={true_a1}, a2={true_a2}, b={true_b}, bias={true_bias}")
print(f"Generated time series x: {x[:10]}...")  # Show first 10 points

# ============================================================================
# Construct least squares problem: Ax = b
# ============================================================================
print("\n" + "-" * 70)
print("Step 1: Construct regression matrix A and target vector b")
print("-" * 70)

order = 2  # Model order
A = []  # Feature matrix
b = []  # Target vector

# Start from t=2 (need history t-1 and t-2)
for t in range(order, T):
    # Construct feature vector: [x[t-1], x[t-2], u[t], 1]
    feature_vector = [
        x[t-1],      # Lag term 1
        x[t-2],      # Lag term 2
        u[t],        # Input term
        1.0          # Bias term
    ]
    A.append(feature_vector)
    b.append(x[t])  # Target value

A = np.array(A)
b = np.array(b)

print(f"\n矩阵 A 的形状: {A.shape} (样本数 × 特征数)")
print(f"向量 b 的形状: {b.shape}")

print("\n矩阵 A 的前 5 行:")
print("     x[t-1]    x[t-2]    u[t]     bias")
for i in range(min(5, len(A))):
    print(f"t={i+2}: {A[i]}")

print(f"\n向量 b 的前 5 个元素 (对应 x[t]):")
print(b[:5])

# ============================================================================
# 使用最小二乘法求解
# ============================================================================
print("\n" + "-" * 70)
print("步骤 2: 使用 np.linalg.lstsq 求解 min ||Ax - b||²")
print("-" * 70)

# 这就是 DEConfig.py:136 行的核心操作！
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print(f"\n求解得到的参数:")
print(f"  a1 (x[t-1]的系数) = {params[0]:.6f}  (真实值: {true_a1})")
print(f"  a2 (x[t-2]的系数) = {params[1]:.6f}  (真实值: {true_a2})")
print(f"  b  (u[t]的系数)   = {params[2]:.6f}  (真实值: {true_b})")
print(f"  bias (偏置项)     = {params[3]:.6f}  (真实值: {true_bias})")

# 计算拟合误差
predictions = A @ params
errors = np.abs(predictions - b)
max_error = np.max(errors)
mean_error = np.mean(errors)

print(f"\n拟合误差:")
print(f"  最大误差: {max_error:.6f}")
print(f"  平均误差: {mean_error:.6f}")

# ============================================================================
# 示例 2: 多变量系统（类似 Duffing 振子）
# ============================================================================
print("\n\n" + "=" * 70)
print("示例 2: 二维非线性系统 (模拟 Duffing 振子)")
print("=" * 70)

# 真实系统:
# x1[t] = 0.9*x1[t-1] + 0.5*x2[t-1] + 0.1
# x2[t] = -0.3*x1[t-1] + 0.8*x2[t-1] + 0.2*x1[t-1]^2 + 0.05

T = 30
x1 = np.zeros(T)
x2 = np.zeros(T)
u = np.zeros(T)  # 无外部输入

# 初始条件
x1[0] = 1.0
x2[0] = 0.5

# 生成时间序列
for t in range(1, T):
    x1[t] = 0.9*x1[t-1] + 0.5*x2[t-1] + 0.1
    x2[t] = -0.3*x1[t-1] + 0.8*x2[t-1] + 0.2*x1[t-1]**2 + 0.05
    x1[t] += np.random.randn() * 0.005
    x2[t] += np.random.randn() * 0.005

print(f"\n生成的时间序列:")
print(f"x1: {x1[:8]}...")
print(f"x2: {x2[:8]}...")

# ============================================================================
# 为每个变量分别构造回归问题
# ============================================================================
print("\n" + "-" * 70)
print("步骤 1: 为 x1 构造回归矩阵（包含非线性项 x1^2）")
print("-" * 70)

order = 1
A1 = []
b1 = []

for t in range(order, T):
    # 特征向量: [x1[t-1], x2[t-1], x1[t-1]^2, u[t], 1]
    feature_vector = [
        x1[t-1],           # 线性滞后项
        x2[t-1],           # 交叉项
        x1[t-1]**2,        # 非线性项 (这就是 get_items 中 fun_list 的作用)
        u[t],              # 输入项
        1.0                # 偏置项
    ]
    A1.append(feature_vector)
    b1.append(x1[t])

A1 = np.array(A1)
b1 = np.array(b1)

print(f"\n矩阵 A1 的形状: {A1.shape}")
print("\n矩阵 A1 的前 3 行:")
print("     x1[t-1]   x2[t-1]   x1²[t-1]  u[t]     bias")
for i in range(min(3, len(A1))):
    print(f"t={i+1}: {A1[i]}")

# 求解 x1 的参数
params1 = np.linalg.lstsq(A1, b1, rcond=None)[0]
print(f"\nx1 的拟合参数:")
print(f"  系数(x1[t-1]) = {params1[0]:.6f}  (真实: 0.9)")
print(f"  系数(x2[t-1]) = {params1[1]:.6f}  (真实: 0.5)")
print(f"  系数(x1²)     = {params1[2]:.6f}  (真实: 0.0)")
print(f"  系数(u[t])    = {params1[3]:.6f}  (真实: 0.0)")
print(f"  偏置项        = {params1[4]:.6f}  (真实: 0.1)")

# ============================================================================
print("\n" + "-" * 70)
print("步骤 2: 为 x2 构造回归矩阵")
print("-" * 70)

A2 = []
b2 = []

for t in range(order, T):
    feature_vector = [
        x1[t-1],
        x2[t-1],
        x1[t-1]**2,
        u[t],
        1.0
    ]
    A2.append(feature_vector)
    b2.append(x2[t])

A2 = np.array(A2)
b2 = np.array(b2)

# 求解 x2 的参数
params2 = np.linalg.lstsq(A2, b2, rcond=None)[0]
print(f"\nx2 的拟合参数:")
print(f"  系数(x1[t-1]) = {params2[0]:.6f}  (真实: -0.3)")
print(f"  系数(x2[t-1]) = {params2[1]:.6f}  (真实: 0.8)")
print(f"  系数(x1²)     = {params2[2]:.6f}  (真实: 0.2)")
print(f"  系数(u[t])    = {params2[3]:.6f}  (真实: 0.0)")
print(f"  偏置项        = {params2[4]:.6f}  (真实: 0.05)")

# 计算拟合误差
pred1 = A1 @ params1
pred2 = A2 @ params2
err1 = np.max(np.abs(pred1 - b1))
err2 = np.max(np.abs(pred2 - b2))

print(f"\n拟合误差:")
print(f"  x1 最大误差: {err1:.6f}")
print(f"  x2 最大误差: {err2:.6f}")

# ============================================================================
# 示例 3: 模拟 DEConfig.py 中的完整流程
# ============================================================================
print("\n\n" + "=" * 70)
print("示例 3: 模拟 DEConfig.py 的 work_normal 函数")
print("=" * 70)

def simulate_append_data(data, input_data, order):
    """
    模拟 DEConfig.py 中的 append_data 函数
    
    参数:
        data: shape (n_vars, n_timesteps) - 状态变量时间序列
        input_data: shape (n_inputs, n_timesteps) - 输入信号
        order: 模型阶数
    
    返回:
        matrix_list: 每个变量的特征矩阵列表
        b_list: 每个变量的目标向量列表
    """
    n_vars = data.shape[0]
    n_timesteps = data.shape[1]
    
    matrix_list = [[] for _ in range(n_vars)]
    b_list = [[] for _ in range(n_vars)]
    
    # 滑动窗口遍历时间序列
    for i in range(n_timesteps - order):
        # 提取滞后项 (倒序: t-1, t-2, ..., t-order)
        if i == 0:
            this_line = data[:, (order - 1)::-1]  # 从 order-1 到 0
            this_line_input = input_data[:, order::-1]
        else:
            this_line = data[:, (order + i - 1):(i - 1):-1]
            this_line_input = input_data[:, (order + i):(i - 1):-1]
        
        # 为每个变量构造特征向量
        for var_idx in range(n_vars):
            # 特征向量: [x_var[t-1], x_var[t-2], ..., u[t], bias]
            feature_vector = []
            
            # 添加该变量的滞后项
            for lag in range(order):
                feature_vector.append(this_line[var_idx, lag])
            
            # 添加输入项
            feature_vector.append(this_line_input[0, 0])
            
            # 添加偏置项
            feature_vector.append(1.0)
            
            matrix_list[var_idx].append(feature_vector)
            b_list[var_idx].append(data[var_idx, i + order])
    
    return matrix_list, b_list


def simulate_work_normal(data, input_data, order):
    """
    模拟 DEConfig.py 中的 work_normal 函数
    """
    # 构造回归矩阵
    matrix_list, b_list = simulate_append_data(data, input_data, order)
    
    res = []
    err = []
    
    # 对每个变量求解最小二乘
    for A, b in zip(matrix_list, b_list):
        A = np.array(A)
        b = np.array(b)
        
        # 核心操作: 最小二乘求解
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # 计算拟合误差
        max_err = np.max(np.abs((A @ x) - b))
        
        res.append(x)
        err.append(max_err)
    
    return res, err


# 准备测试数据
data = np.array([x1, x2])  # shape: (2, 30)
input_data = np.array([u])  # shape: (1, 30)

print(f"\n输入数据形状:")
print(f"  data: {data.shape} (变量数 × 时间步数)")
print(f"  input_data: {input_data.shape}")

# 执行拟合
order = 1
params_list, errors = simulate_work_normal(data, input_data, order)

print(f"\n拟合结果:")
for i, (params, err) in enumerate(zip(params_list, errors)):
    print(f"\n变量 x{i+1}:")
    print(f"  参数向量: {params}")
    print(f"  最大拟合误差: {err:.6f}")

# ============================================================================
# 可视化结果
# ============================================================================
print("\n\n" + "=" * 70)
print("生成可视化图表...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 示例1的拟合结果
ax = axes[0, 0]
t_plot = np.arange(order, T)
ax.plot(t_plot, b, 'o-', label='真实值', markersize=4)
ax.plot(t_plot, predictions, 's--', label='拟合值', markersize=4)
ax.set_xlabel('时间步 t')
ax.set_ylabel('x[t]')
ax.set_title('示例1: 一维 AR(2) 模型拟合')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2: 示例1的拟合误差
ax = axes[0, 1]
ax.plot(t_plot, errors, 'r-', linewidth=2)
ax.axhline(y=mean_error, color='b', linestyle='--', label=f'平均误差: {mean_error:.4f}')
ax.set_xlabel('时间步 t')
ax.set_ylabel('绝对误差')
ax.set_title('示例1: 拟合误差')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: 示例2的 x1 拟合
ax = axes[1, 0]
t_plot2 = np.arange(order, T)
ax.plot(t_plot2, b1, 'o-', label='真实 x1', markersize=4)
ax.plot(t_plot2, pred1, 's--', label='拟合 x1', markersize=4)
ax.set_xlabel('时间步 t')
ax.set_ylabel('x1[t]')
ax.set_title('示例2: x1 变量拟合（含非线性项）')
ax.legend()
ax.grid(True, alpha=0.3)

# 图4: 示例2的 x2 拟合
ax = axes[1, 1]
ax.plot(t_plot2, b2, 'o-', label='真实 x2', markersize=4)
ax.plot(t_plot2, pred2, 's--', label='拟合 x2', markersize=4)
ax.set_xlabel('时间步 t')
ax.set_ylabel('x2[t]')
ax.set_title('示例2: x2 变量拟合（含非线性项）')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('result/least_squares_demo.png', dpi=150, bbox_inches='tight')
print("\n图表已保存到: result/least_squares_demo.png")

# ============================================================================
# 总结
# ============================================================================
print("\n\n" + "=" * 70)
print("核心要点总结")
print("=" * 70)
print("""
1. np.linalg.lstsq(A, b, rcond=None)[0] 的作用:
   - 求解最优参数 x，使得 ||Ax - b||² 最小
   - 这是标准的最小二乘法，适用于过定方程组（样本数 > 特征数）

2. 在 NARX 模型中的应用:
   - A 的每一行 = 一个时间点的特征向量 [x[t-1], x[t-2], ..., u[t], 1]
   - b 的每个元素 = 对应时间点的目标值 x[t]
   - 求解得到的 x = [a1, a2, ..., b_input, bias]

3. DEConfig.py 中的实现:
   - append_data(): 构造矩阵 A 和向量 b
   - work_normal(): 对每个变量调用 lstsq 求解
   - get_items(): 构造单个特征向量（支持非线性项）

4. 为什么这个方法有效:
   - 将时间序列预测转化为线性回归问题
   - 可以处理多变量、高阶、非线性系统
   - 通过拟合误差判断模式切换（核心创新）
""")

print("\n运行完成！")

