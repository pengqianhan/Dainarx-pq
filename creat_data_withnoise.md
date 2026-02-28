# CreatData 添加测量噪声 — 改动记录

## 噪声方案

采用**按维度标准差缩放**的测量噪声：

```
x̃[dim] = x[dim] + N(0, noise_level × std(x[dim]))
```

- `noise_level` 表示噪声占该维度信号波动幅度的比例（e.g., 0.05 = 5%）
- 每个状态维度独立计算 std 并独立加噪
- 当某维度 std=0 时，退化为绝对噪声 `N(0, noise_level)`
- 噪声仅叠加到 `state_data`，不影响 `mode_data` 和 `change_points`

## 改动清单

| 文件 | 位置 | 改动内容 |
|------|------|----------|
| `CreatData.py` | L10 | `creat_data` 函数新增参数 `noise_level: float = 0.0` |
| `CreatData.py` | L59-67 | 在 `state_data` 保存前，按维度 std 缩放生成高斯噪声并叠加 |
| `main.py` | L77 | `main()` 函数新增参数 `noise_level: float = 0.0` 并传递给 `creat_data` |
| `main.py` | L169 | 示例调用改为 `noise_level=0.05`（5% 噪声） |

## 使用方式

```python
# 无噪声（默认，与原始行为一致）
creat_data('automata/1.json', 'data', 0.01, 10)

# 5% 测量噪声
creat_data('automata/1.json', 'data', 0.01, 10, noise_level=0.05)

# 通过 main 函数调用
main("./automata/non_linear/duffing.json", noise_level=0.1)
```

## 方案选型说明

| 方案 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| 绝对噪声 | `x̃ = x + N(0, σ)` | 简单 | 对小值变量噪声过大，对大值变量噪声过小 |
| 纯乘性噪声 | `x̃ = x · (1 + N(0, σ))` | 自动缩放 | x ≈ 0 时噪声消失 |
| **按维度 std 缩放** | `x̃ = x + N(0, σ · std(x_dim))` | 每个维度独立缩放，x ≈ 0 时仍有噪声 | 需先生成完整轨迹 |
