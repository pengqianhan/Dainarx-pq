我来结合论文详细解释`find_change_point`函数的流程。首先让我查看相关的代码文件来更好地理解整个实现。

[2 tools called]

现在让我查看更多相关的代码来理解完整的流程：

[2 tools called]

现在我来详细解释`find_change_point`函数的流程，结合论文中的理论基础：

## `find_change_point` 函数流程详解

### 理论背景
根据论文第4.2节"Trace Segmentation"，该函数实现了**变点检测(Changepoint Detection)**，这是DAINARX方法的第一步。其目标是将离散时间轨迹分割成若干段，使得相邻段由不同的模态动力学生成。

### 核心概念：变点属性(Changepoints Property)
论文中定义的变点属性(Property 9)：
- 给定离散时间轨迹 ξ 和模板NARX模型 N
- 变点集合 CP = {p₀, p₁, ..., pₛ} 满足：
  1. 每个子段 ξₚᵢ,ₚᵢ₊₁ 都可以被NARX模型拟合
  2. 扩展的子段 ξₚᵢ,(ₚᵢ₊₁)₊₁ 不能被NARX模型拟合

### 函数详细流程

#### 1. **初始化阶段** (第25-33行)
```python
change_points = []      # 存储检测到的变点
error_datas = []       # 存储错误数据(未使用)
tail_len = 0           # 尾部长度计数器
pos = 0                # 当前滑动窗口位置
last = None            # 上一次的拟合阶数
merge_th = w if merge_th is None else merge_th  # 合并阈值
eps = get_feature.get_eps(data)  # 获取误差阈值
```

**`eps`的作用**：根据`DEConfig.py`第78行，`eps = 1e-6 * dt * max(data)`，这是基于数据规模的自适应误差阈值。

#### 2. **滑动窗口检测** (第35-43行)
```python
while pos + w < data.shape[1]:
    # 在当前窗口内进行NARX模型拟合
    feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)], 
                                            input_data[:, pos:(pos + w)])
```

**关键操作**：
- 使用大小为`w`的滑动窗口
- 对每个窗口调用`get_feature`进行NARX模型拟合
- 返回：特征参数(`feature`)、拟合误差(`now_err`)、拟合阶数(`fit_order`)

#### 3. **变点判定逻辑** (第37-42行)
```python
if last is not None:
    if (max(now_err) > eps) and tail_len == 0:
        change_points.append(pos + w - 1)  # 记录变点位置
        tail_len = w                       # 设置尾部保护长度
    tail_len = max(tail_len - 1, 0)      # 递减尾部计数器
```

**判定条件**：
- `max(now_err) > eps`：当前窗口的最大拟合误差超过阈值
- `tail_len == 0`：不在尾部保护期内

**尾部保护机制**：
- 检测到变点后，设置`tail_len = w`
- 在接下来的`w`个时间步内不再检测新变点
- 避免在同一个真实变点附近检测到多个虚假变点

#### 4. **变点合并** (第45-48行)
```python
res = mergeChangePoints(change_points, merge_th)
res.append(data.shape[1])  # 添加轨迹终点
res.insert(0, 0)           # 添加轨迹起点
```

**`mergeChangePoints`函数**：
- 将距离小于`merge_th`的变点合并
- 消除由噪声引起的虚假变点
- 确保变点间有足够的间隔

### 算法核心思想

#### NARX模型拟合检测
函数通过连续的NARX模型拟合来检测模态切换：
1. **正常段落**：NARX模型能够很好地拟合数据，误差较小
2. **模态切换点**：由于动力学发生变化，NARX模型拟合误差突然增大
3. **变点标记**：当误差超过阈值时，标记为潜在变点

#### 与论文理论的对应关系
- **滑动窗口**对应论文中的轨迹段 ξₚᵢ,ₚᵢ₊₁
- **NARX拟合**对应论文中"可被N拟合"的判定条件
- **误差阈值检测**实现了论文中变点属性的条件(ii)

### 参数说明
- `w=10`：滑动窗口大小，平衡检测精度和计算效率
- `merge_th`：变点合并阈值，默认等于窗口大小
- `eps`：自适应误差阈值，基于数据规模动态调整

### 输出结果
返回的`res`是一个包含所有变点的列表，其中：
- `res[0] = 0`：轨迹起点
- `res[-1] = data.shape[1]`：轨迹终点  
- 中间元素为检测到的变点位置

这个变点列表将用于后续的轨迹分割，为下一步的段聚类(Segment Clustering)提供基础。