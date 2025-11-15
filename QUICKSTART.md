# 快速开始指南 - Dainarx LLM 自动配置

## 5分钟快速上手

### 1. 安装依赖 (30秒)

```bash
pip install numpy scipy google-generativeai python-dotenv matplotlib scikit-learn networkx
```

### 2. 配置 API 密钥 (1分钟)

编辑 `.env` 文件:
```bash
GEMINI_API_KEY=your_api_key_here
```

获取密钥: https://makersuite.google.com/app/apikey

### 3. 验证安装 (30秒)

```bash
python demos/test_installation.py
```

预期输出:
```
✓ All tests passed! Installation is ready.
```

### 4. 运行 Duffing 演示 (2-5分钟)

```bash
python demos/duffing_auto_config.py --test --pop-size 5 --generations 3
```

## 输出示例

```
=== LLM 分析结果 ===
系统类型: 非线性振荡器
推荐阶数: [2, 3]
非线性项: ["x[?]**3"]
需要重置: True

=== 遗传算法优化 ===
Generation 3/3: Best fitness = 0.87

=== 最优配置 ===
  order: 2
  other_items: x[?]**3
  window_size: 10
  kernel: rbf
  need_reset: True

结果保存到: result/duffing_auto_config_result.json
```

## 核心文件说明

```
llm_config_optimizer/        # 核心模块
├── data_analyzer.py         # 数据特征提取
├── llm_analyzer.py          # LLM 配置分析
├── genetic_optimizer.py     # 遗传算法优化
└── fitness_evaluator.py     # 适应度评估

demos/
└── duffing_auto_config.py   # Duffing 系统演示

LearningNotes/
├── llm_genetic_config_design.md   # 详细设计文档
└── limitation.md            # 算法限制分析

PROJECT_SUMMARY.md           # 完整项目总结
```

## 命令选项

```bash
# 快速测试 (2-5 分钟)
python demos/duffing_auto_config.py --test --pop-size 5 --generations 3

# 完整运行 (15-30 分钟)
python demos/duffing_auto_config.py --full-run --pop-size 20 --generations 10

# 跳过 LLM 分析
python demos/duffing_auto_config.py --no-llm --pop-size 10 --generations 5
```

## 下一步

1. 查看 `PROJECT_SUMMARY.md` 了解完整架构
2. 阅读 `llm_config_optimizer/README.md` 了解模块详情
3. 参考 `demos/README.md` 了解如何应用到其他系统
4. 查看 `LearningNotes/llm_genetic_config_design.md` 了解设计细节

## 常见问题

**Q: API 密钥错误?**
A: 检查 `.env` 文件中的 `GEMINI_API_KEY` 是否正确

**Q: 适应度分数低?**
A: 增大种群规模 (`--pop-size 20`) 和代数 (`--generations 10`)

**Q: 运行时间太长?**
A: 使用 `--test` 模式和较小的种群规模

**Q: 如何应用到其他系统?**
A: 修改 `demos/duffing_auto_config.py` 中的 `json_path` 变量

## 帮助

```bash
python demos/duffing_auto_config.py --help
```

详细文档: `PROJECT_SUMMARY.md`
