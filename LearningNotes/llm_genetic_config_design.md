# LLM + 遗传算法配置优化方案

## 1. 问题分析

根据 `limitation.md`，Dainarx 算法存在以下关键配置依赖性限制：

### 1.1 关键参数（需要人工设定）
- **NARX 阶数 (order)**: 影响整个流程，决定历史依赖深度
- **非线性项 (other_items)**: 需要领域知识，如 `x[?]**3` for Duffing
- **窗口大小 (window_size)**: 影响变点检测质量
- **SVM 超参数**: C, kernel, class_weight
- **自环 (self_loop)**: 是否允许同一模式连续出现
- **重置函数 (need_reset)**: 是否学习状态跳变

## 2. 解决方案架构

### 2.1 总体思路
使用 **LLM 作为智能分析器** + **遗传算法作为优化引擎**：

```
输入: 时间序列数据 + 系统描述
  ↓
LLM 分析阶段: 推断系统特性
  - 系统类型识别（线性/非线性）
  - 非线性项建议
  - 参数范围建议
  ↓
遗传算法优化阶段: 搜索最优配置
  - 种群初始化（基于 LLM 建议）
  - 适应度评估（运行 Dainarx + 评估指标）
  - 进化迭代（选择、交叉、变异）
  ↓
输出: 最优配置参数
```

### 2.2 核心组件

#### 2.2.1 LLM 配置分析器 (`LLMConfigAnalyzer`)
**职责**: 分析系统特性，提供初始配置建议

**输入**:
- 时间序列数据的统计特征
- 系统描述（可选）
- Ground truth 自动机信息（用于验证）

**输出**:
```python
{
    "system_type": "nonlinear_oscillator",
    "suggested_order": [2, 3, 4],  # 建议的阶数范围
    "nonlinear_terms": ["x[?]**3", "x[?]**2"],  # 建议的非线性项
    "window_size_range": [8, 15],
    "has_state_jumps": True,  # 是否有重置
    "allows_self_loop": False,
    "svm_kernel": "rbf",
    "reasoning": "基于数据的频谱分析和振幅特性，系统呈现典型的Duffing振子行为..."
}
```

**LLM Prompt 设计**:
```
你是一个混合系统建模专家。给定以下时间序列数据特征：
- 数据维度: {dim}
- 采样率: {dt}
- 数据统计: {stats}
- 频谱特性: {fft_features}
- 相空间轨迹: {phase_portrait}

请分析：
1. 系统类型（线性/非线性/混合）
2. 建议的NARX阶数范围（2-5）
3. 可能的非线性项（如 x**2, x**3, sin(x)等）
4. 是否存在状态跳变
5. 推荐的变点检测窗口大小

输出JSON格式...
```

#### 2.2.2 遗传算法优化器 (`GeneticOptimizer`)
**职责**: 在参数空间中搜索最优配置

**染色体编码**:
```python
chromosome = {
    "order": int,           # [2, 3, 4, 5]
    "window_size": int,     # [5, 20]
    "other_items": str,     # 从候选列表选择
    "kernel": str,          # ['linear', 'rbf', 'poly']
    "svm_c": float,         # [1e2, 1e8] (log scale)
    "class_weight": float,  # [1, 100]
    "self_loop": bool,
    "need_reset": bool
}
```

**适应度函数**:
```python
def fitness(config, data, ground_truth):
    # 运行 Dainarx
    sys, slice_data = run_dainarx(data, config)

    # 多目标评估
    scores = {
        "mode_accuracy": calculate_mode_accuracy(slice_data, gt),
        "chp_f1": calculate_changepoint_f1(slice_data, gt),
        "fitting_error": calculate_fitting_error(sys, data),
        "complexity_penalty": len(slice_data) * 0.01  # 偏好简单模型
    }

    # 加权综合得分
    fitness = 0.4 * mode_accuracy + 0.3 * chp_f1 - 0.2 * fitting_error - 0.1 * complexity_penalty
    return fitness
```

**遗传算子**:
- **选择**: 锦标赛选择 (tournament size=3)
- **交叉**: 单点交叉 + 语义保留（如 other_items 不能随意组合）
- **变异**: 自适应变异率（初期高，后期低）

#### 2.2.3 数据特征提取器 (`DataAnalyzer`)
**职责**: 从原始数据提取用于 LLM 分析的特征

**提取特征**:
```python
features = {
    "dimension": int,
    "sample_rate": float,
    "duration": float,
    "statistics": {
        "mean": [...],
        "std": [...],
        "min": [...],
        "max": [...],
        "skewness": [...],
        "kurtosis": [...]
    },
    "spectral": {
        "dominant_frequencies": [...],
        "power_spectrum": [...]
    },
    "dynamics": {
        "lyapunov_exponent": float,  # 混沌度
        "correlation_dimension": float
    },
    "transitions": {
        "rough_changepoint_count": int,  # 粗略检测
        "avg_segment_length": float
    }
}
```

## 3. 实现细节

### 3.1 目录结构
```
Dainarx-pq/
├── llm_config_optimizer/
│   ├── __init__.py
│   ├── llm_analyzer.py       # LLM 配置分析器
│   ├── genetic_optimizer.py  # 遗传算法优化器
│   ├── data_analyzer.py      # 数据特征提取
│   ├── fitness_evaluator.py  # 适应度评估
│   └── prompts/
│       ├── system_analysis.txt
│       └── config_suggestion.txt
├── demos/
│   └── duffing_auto_config.py  # Duffing 系统演示
└── .env  # API keys
```

### 3.2 LLM 集成 (参考 llmlex)

**API 调用封装**:
```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

class GeminiClient:
    def __init__(self, model_name="models/gemini-flash-lite-latest"):
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def analyze_system(self, data_features, prompt_template):
        prompt = prompt_template.format(**data_features)
        response = self.model.generate_content(prompt)
        return self.parse_json_response(response.text)

    def parse_json_response(self, text):
        # 提取 JSON（处理 markdown 代码块）
        import re
        import json
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
```

### 3.3 遗传算法流程

```python
class GeneticOptimizer:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def optimize(self, data, llm_suggestions):
        # 1. 初始化种群（50% 基于 LLM 建议，50% 随机）
        population = self.initialize_population(llm_suggestions)

        best_config = None
        best_fitness = -float('inf')

        for gen in range(self.generations):
            # 2. 评估适应度（并行执行）
            fitness_scores = [self.evaluate(ind, data) for ind in population]

            # 3. 记录最佳
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_config = population[max_idx]

            print(f"Gen {gen}: Best fitness = {best_fitness}")

            # 4. 选择
            parents = self.tournament_selection(population, fitness_scores)

            # 5. 交叉和变异
            offspring = self.crossover_and_mutate(parents)

            # 6. 替换（精英保留）
            population = self.elitism_replacement(population, offspring, fitness_scores)

        return best_config, best_fitness

    def evaluate(self, config, data):
        try:
            # 运行 Dainarx
            sys, slice_data = run_dainarx_with_config(data, config)
            # 计算适应度
            return calculate_fitness(sys, slice_data, data)
        except Exception as e:
            print(f"Config failed: {e}")
            return -1e6  # 惩罚非法配置
```

## 4. Duffing 系统 Demo 流程

### 4.1 输入
- Duffing 自动机规范: `automata/non_linear/duffing.json`
- 生成的时间序列数据

### 4.2 执行流程
```python
# 1. 加载数据
data = load_duffing_data()

# 2. 数据分析
analyzer = DataAnalyzer()
features = analyzer.extract_features(data)

# 3. LLM 分析
llm = GeminiClient(model="models/gemini-flash-lite-latest")  # 测试
suggestions = llm.analyze_system(features)
print("LLM Suggestions:", suggestions)

# 4. 遗传算法优化
ga = GeneticOptimizer(population_size=20, generations=10)
best_config, fitness = ga.optimize(data, suggestions)

# 5. 验证
print("Best Config:", best_config)
print("Fitness:", fitness)

# 6. 与人工配置对比
manual_config = load_manual_config()
comparison = compare_results(best_config, manual_config, data)
print("Comparison:", comparison)
```

### 4.3 预期输出
```
=== LLM 分析结果 ===
系统类型: 非线性振荡器
推荐阶数: [2, 3]
非线性项: ["x[?]**3"]
窗口大小: [10, 12]
需要重置: True
SVM 核: rbf

=== 遗传算法优化 ===
Generation 0: Best fitness = 0.65
Generation 1: Best fitness = 0.72
...
Generation 10: Best fitness = 0.89

=== 最优配置 ===
{
  "order": 2,
  "window_size": 10,
  "other_items": "x[?]**3",
  "kernel": "rbf",
  "svm_c": 1e6,
  "class_weight": 50,
  "self_loop": False,
  "need_reset": True
}

=== 与人工配置对比 ===
                    | 自动配置 | 人工配置 | 差异
Mode Accuracy       | 0.95     | 0.92     | +3%
ChangePoint F1      | 0.88     | 0.85     | +3%
Fitting Error       | 0.012    | 0.015    | -20%
```

## 5. 优化策略

### 5.1 两阶段优化
- **阶段1**: LLM 粗搜索（快速收敛）
- **阶段2**: GA 精细调优（局部优化）

### 5.2 缓存机制
- 缓存已评估的配置，避免重复计算
- 使用配置哈希作为键

### 5.3 并行评估
- 利用多进程并行评估适应度
- 加速遗传算法迭代

### 5.4 早停机制
- 若连续 N 代无改进，提前停止
- 节省计算资源

## 6. 扩展性

### 6.1 支持其他系统
- 只需提供新系统的数据和描述
- LLM 自动适配分析

### 6.2 配置模板库
- 保存成功的配置作为模板
- 用于相似系统的快速启动

### 6.3 增量学习
- 基于历史优化结果改进 LLM prompt
- 提高后续分析准确性

## 7. 技术栈

- **LLM**: Google Gemini (flash-lite for testing, flash for production)
- **遗传算法**: DEAP 或自实现
- **数据分析**: NumPy, SciPy
- **并行**: multiprocessing
- **配置管理**: python-dotenv

## 8. 成功指标

- **自动化率**: 90% 的系统无需人工干预
- **准确性**: 自动配置达到人工配置 95% 的性能
- **效率**: 优化时间 < 10 分钟（20 代遗传算法）
- **鲁棒性**: 对不同系统类型都有合理建议
