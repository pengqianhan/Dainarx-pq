import numpy as np

# 创建一个示例数据矩阵
data = np.array([
    [1, 2, 3, 4, 5],    # 第0行
    [6, 7, 8, 9, 10],   # 第1行
    [11, 12, 13, 14, 15], # 第2行
    [16, 17, 18, 19, 20], # 第3行
    [21, 22, 23, 24, 25]  # 第4行
])

print("Original data:")
print(data)
print()

# Example 1: bias = 3, i = 2
bias = 3
i = 2
result = data[i, (bias - 1)::-1]
print(f"data[{i}, ({bias} - 1)::-1] = data[{i}, {bias-1}::-1]")
print(f"Result: {result}")
print(f"Explanation: Take values from column {bias-1} in row {i} in reverse order")
print()

# Example 2: bias = 4, i = 1
bias = 4
i = 1
result = data[i, (bias - 1)::-1]
print(f"data[{i}, ({bias} - 1)::-1] = data[{i}, {bias-1}::-1]")
print(f"Result: {result}")
print(f"Explanation: Take values from column {bias-1} in row {i} in reverse order")
print()

# Example 3: bias = 2, i = 3
bias = 2
i = 3
result = data[i, (bias - 1)::-1]
print(f"data[{i}, ({bias} - 1)::-1] = data[{i}, {bias-1}::-1]")
print(f"Result: {result}")
print(f"Explanation: Take values from column {bias-1} in row {i} in reverse order")
print()

# Compare with forward slice
bias = 3
i = 2
normal_slice = data[i, :(bias)]  # Forward slice, from start to bias column
print(f"Forward slice data[{i}, :{bias}] = {normal_slice}")
print(f"Reverse slice data[{i}, ({bias}-1)::-1] = {data[i, (bias - 1)::-1]}")
