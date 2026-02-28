import numpy as np
import matplotlib.pyplot as plt

# 设置系统参数
time_steps = 50       # 模拟的时间步数
a = 0.95              # 系统动态系数 (随时间衰减)
x0 = 10.0             # 初始状态
sigma = 0.5           # 噪声的标准差 (高斯分布)

# ==========================================
# 1. 纯净数据 (无噪声)
# ==========================================
x_clean = np.zeros(time_steps)
x_clean[0] = x0
for t in range(1, time_steps):
    x_clean[t] = a * x_clean[t-1]

# ==========================================
# 2. 测量噪声 (Measurement Noise)
# 解释：噪声直接叠加在最终测量到的纯净数据上
# ==========================================
# 生成服从高斯分布的测量噪声
meas_noise = np.random.normal(loc=0.0, scale=sigma, size=time_steps)
x_meas_noisy = x_clean + meas_noise

# ==========================================
# 3. 过程噪声 (Process Noise)
# 解释：噪声在系统的演化过程中每一步都会加入，从而影响未来的状态
# ==========================================
x_proc_noisy = np.zeros(time_steps)
x_proc_noisy[0] = x0
for t in range(1, time_steps):
    # 生成单步的高斯过程噪声
    proc_noise_step = np.random.normal(loc=0.0, scale=sigma)
    # 噪声作用于系统动态（微分方程/差分方程）中
    x_proc_noisy[t] = a * x_proc_noisy[t-1] + proc_noise_step

# ==========================================
# 可视化对比
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(x_clean, label='Clean Data', color='green', linewidth=2)
plt.plot(x_meas_noisy, label='With Measurement Noise', color='blue', alpha=0.7, linestyle='--')
plt.plot(x_proc_noisy, label='With Process Noise', color='red', alpha=0.7, linestyle='-.')
plt.title('Comparison of Measurement Noise vs Process Noise')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.grid(True)
plt.savefig('noise_learn.png')