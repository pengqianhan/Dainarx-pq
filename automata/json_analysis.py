#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计automata目录下所有JSON文件中"input"字段的数量
"""

import json
import os
from pathlib import Path


def count_input_in_json(json_file_path):
    """
    统计单个JSON文件中"input"字段的出现次数

    Args:
        json_file_path: JSON文件路径

    Returns:
        int: "input"字段的数量
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查automaton中是否有input字段
        if 'automaton' in data and 'input' in data['automaton']:
            return 1
        return 0
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return 0


def count_mode_in_json(json_file_path):
    """
    统计单个JSON文件中automaton.mode列表的长度

    Args:
        json_file_path: JSON文件路径

    Returns:
        int: mode列表的长度，如果不存在则返回0
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查automaton中是否有mode字段
        if 'automaton' in data and 'mode' in data['automaton']:
            mode_list = data['automaton']['mode']
            if isinstance(mode_list, list):
                return len(mode_list)
        return 0
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return 0


def analyze_config_in_json(json_file_path):
    """
    分析单个JSON文件中config字段的配置项

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含配置信息的字典，如果没有config则返回None
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有config字段
        if 'config' in data:
            config = data['config']
            result = {
                'order': config.get('order'),
                'need_reset': config.get('need_reset'),
                'kernel': config.get('kernel'),
                'total_time': config.get('total_time'),
                'all_keys': list(config.keys())
            }
            return result
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None


def get_total_time_stats(config_stats):
    """
    从config统计信息中提取total_time的统计数据

    Args:
        config_stats: 包含所有文件config信息的字典

    Returns:
        dict: total_time统计信息，包含:
            - value_distribution: 按值分组的文件列表
            - min_value: 最小值
            - max_value: 最大值
            - avg_value: 平均值
            - files_with_total_time: 有total_time字段的文件数
            - files_without_total_time: 没有total_time字段的文件数
    """
    total_time_stats = {}
    values = []

    for file_path, config in config_stats.items():
        total_time_val = config.get('total_time')
        if total_time_val not in total_time_stats:
            total_time_stats[total_time_val] = []
        total_time_stats[total_time_val].append(file_path)

        if total_time_val is not None:
            values.append(total_time_val)

    result = {
        'value_distribution': total_time_stats,
        'files_with_total_time': len(values),
        'files_without_total_time': len(config_stats) - len(values),
    }

    if values:
        result['min_value'] = min(values)
        result['max_value'] = max(values)
        result['avg_value'] = sum(values) / len(values)
    else:
        result['min_value'] = None
        result['max_value'] = None
        result['avg_value'] = None

    return result


def analyze_edges_in_json(json_file_path):
    """
    分析单个JSON文件中edge字段的条件表达式

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含边和条件信息的字典，如果没有edge则返回None
    """
    import re
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有automaton.edge字段
        if 'automaton' in data and 'edge' in data['automaton']:
            edges = data['automaton']['edge']
            
            edge_info = {
                'edge_count': len(edges),
                'edges': [],
                'conditions': [],
                'has_reset': [],
                'operators': set(),
                'variables': set()
            }
            
            # 定义条件中常见的操作符
            operators_pattern = [
                (r'<=', '<='),
                (r'>=', '>='),
                (r'<(?!=)', '<'),
                (r'>(?!=)', '>'),
                (r'==', '=='),
                (r'!=', '!='),
                (r'\band\b', 'and'),
                (r'\bor\b', 'or'),
                (r'\bnot\b', 'not'),
                (r'abs\s*\(', 'abs()'),
            ]
            
            # 变量模式 (如 x, x1, x2, x[0], x[1] 等)
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[|\s*[<>=!])'
            
            for edge in edges:
                direction = edge.get('direction', 'unknown')
                condition = edge.get('condition', '')
                has_reset = 'reset' in edge
                
                edge_info['edges'].append(direction)
                edge_info['conditions'].append(condition)
                edge_info['has_reset'].append(has_reset)
                
                # 提取操作符
                for pattern, op_name in operators_pattern:
                    if re.search(pattern, condition):
                        edge_info['operators'].add(op_name)
                
                # 提取变量名
                vars_found = re.findall(var_pattern, condition)
                for var in vars_found:
                    # 过滤掉常见的非变量关键词
                    if var not in ['and', 'or', 'not', 'abs', 'True', 'False']:
                        edge_info['variables'].add(var)
            
            # 转换set为list便于JSON序列化
            edge_info['operators'] = list(edge_info['operators'])
            edge_info['variables'] = list(edge_info['variables'])
            
            return edge_info
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None


def extract_equation_terms(equation):
    """
    从方程字符串中提取项（不包含系数）

    Args:
        equation: 方程字符串，例如 "x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3"

    Returns:
        list: 提取出的项列表（不包含系数），例如 ['u', 'x[1]', 'x[0]', 'x[0]**3']
    """
    import re

    # 提取等号右边的部分
    if '=' in equation:
        right_side = equation.split('=', 1)[1].strip()
    else:
        right_side = equation.strip()

    # 按照加减号分割，保留符号
    # 先替换减号为 +- 以便统一处理
    right_side = right_side.replace('-', '+-')

    # 分割成项
    raw_terms = [t.strip() for t in right_side.split('+') if t.strip()]

    normalized_terms = []
    for term in raw_terms:
        # 去除前导的负号（我们只关心项的结构，不关心符号）
        term = term.lstrip('-').strip()
        if not term:
            continue

        # 去除数字系数
        # 模式1: 纯数字开头后跟 * (例如: "0.5 * x[1]" -> "x[1]")
        term = re.sub(r'^[\d.]+\s*\*\s*', '', term)

        # 模式2: 去除空格，标准化
        term = term.replace(' ', '')

        # 去除多余的乘号前的数字系数（如果还有）
        # 例如: "2*x[0]" -> "x[0]"
        term = re.sub(r'^\d+\.?\d*\*', '', term)

        if term and term not in normalized_terms:
            normalized_terms.append(term)

    return sorted(normalized_terms)


def analyze_mode_equations(json_file_path):
    """
    分析单个JSON文件中各个mode的方程项

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含mode方程分析信息的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有automaton.mode字段
        if 'automaton' not in data or 'mode' not in data['automaton']:
            return None

        modes = data['automaton']['mode']
        if not isinstance(modes, list) or len(modes) == 0:
            return None

        mode_info = {
            'mode_count': len(modes),
            'modes': [],
            'all_terms': set(),
            'terms_consistent': True,
            'first_mode_terms': None
        }

        for mode in modes:
            mode_id = mode.get('id', 'unknown')
            equation = mode.get('eq', '')

            if equation:
                terms = extract_equation_terms(equation)
                mode_info['modes'].append({
                    'id': mode_id,
                    'equation': equation,
                    'terms': terms
                })

                # 收集所有项
                mode_info['all_terms'].update(terms)

                # 检查项的一致性
                if mode_info['first_mode_terms'] is None:
                    mode_info['first_mode_terms'] = set(terms)
                else:
                    if set(terms) != mode_info['first_mode_terms']:
                        mode_info['terms_consistent'] = False

        # 转换set为sorted list
        mode_info['all_terms'] = sorted(mode_info['all_terms'])
        mode_info['first_mode_terms'] = sorted(mode_info['first_mode_terms']) if mode_info['first_mode_terms'] else []

        return mode_info

    except Exception as e:
        print(f"Error analyzing mode equations in {json_file_path}: {e}")
        return None


def extract_condition_pattern(condition):
    """
    从条件表达式中提取模式类型

    Args:
        condition: 条件表达式字符串

    Returns:
        list: 识别出的模式类型列表
    """
    patterns = []

    # 简单比较: var op value
    if ' <= ' in condition or ' >= ' in condition or ' < ' in condition or ' > ' in condition:
        patterns.append('comparison')

    # 等式判断
    if ' == ' in condition or ' != ' in condition:
        patterns.append('equality')

    # 复合条件
    if ' and ' in condition:
        patterns.append('compound_and')
    if ' or ' in condition:
        patterns.append('compound_or')

    # 函数调用
    if 'abs(' in condition:
        patterns.append('abs_function')

    # 变量间比较 (如 x1 - x2 < 3)
    import re
    if re.search(r'[a-zA-Z]\d*\s*-\s*[a-zA-Z]\d*', condition):
        patterns.append('var_difference')

    return patterns if patterns else ['simple']


def analyze_variables_in_json(json_file_path):
    """
    分析单个JSON文件中的变量数量和变量名

    通过解析 automaton.var 字段来确定状态变量，
    通过解析 automaton.input 字段来确定输入变量。

    Args:
        json_file_path: JSON文件路径

    Returns:
        dict: 包含变量信息的字典，包括:
            - var_count: 状态变量数量
            - var_names: 状态变量名列表
            - input_count: 输入变量数量
            - input_names: 输入变量名列表
            - total_count: 总变量数量（状态+输入）
        如果没有automaton字段则返回None
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'automaton' not in data:
            return None

        automaton = data['automaton']
        result = {
            'var_count': 0,
            'var_names': [],
            'input_count': 0,
            'input_names': [],
            'total_count': 0
        }

        # 解析状态变量 (var 字段)
        if 'var' in automaton:
            var_str = automaton['var'].strip()
            if ',' in var_str:
                # 多个变量，逗号分隔: "x1, x2, x3"
                var_names = [v.strip() for v in var_str.split(',') if v.strip()]
            else:
                # 单个变量: "x"
                var_names = [var_str]
            result['var_names'] = var_names
            result['var_count'] = len(var_names)

        # 解析输入变量 (input 字段)
        if 'input' in automaton:
            input_str = automaton['input'].strip()
            if ',' in input_str:
                input_names = [v.strip() for v in input_str.split(',') if v.strip()]
            else:
                input_names = [input_str]
            result['input_names'] = input_names
            result['input_count'] = len(input_names)

        result['total_count'] = result['var_count'] + result['input_count']
        return result

    except Exception as e:
        print(f"Error analyzing variables in {json_file_path}: {e}")
        return None


def find_all_json_files(root_dir):
    """
    递归查找所有JSON文件
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        list: 所有JSON文件的路径列表
    """
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def generate_markdown_report(automata_dir, json_files, files_with_input, files_without_input,
                            config_stats, order_stats, need_reset_stats, kernel_stats,
                            all_config_keys, edge_stats=None, mode_stats=None, equation_stats=None,
                            total_time_stats=None, variable_stats=None):
    """
    生成Markdown格式的分析报告

    Args:
        automata_dir: automata目录路径
        json_files: 所有JSON文件列表
        files_with_input: 包含input字段的文件列表
        files_without_input: 不包含input字段的文件列表
        config_stats: config配置统计信息
        order_stats: order参数统计信息
        need_reset_stats: need_reset参数统计信息
        kernel_stats: kernel参数统计信息
        all_config_keys: 所有出现过的config键集合
        edge_stats: edge条件统计信息
        mode_stats: mode数量统计信息
        equation_stats: mode方程分析统计信息
        variable_stats: 变量数量统计信息
    """
    report_path = Path(__file__).parent / "json_analysis_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# JSON Automata Configuration Analysis Report\n\n")
        f.write(f"**扫描目录**: `{automata_dir}`\n\n")
        f.write("---\n\n")

        # 基本统计
        f.write("## 1. 基本统计\n\n")
        f.write(f"- **总JSON文件数**: {len(json_files)}\n")
        f.write(f"- **包含input字段的文件数**: {len(files_with_input)}\n")
        f.write(f"- **不包含input字段的文件数**: {len(files_without_input)}\n")
        f.write(f"- **包含config字段的文件数**: {len(config_stats)}\n\n")

        # Input字段分析
        f.write("## 2. Input字段分析\n\n")
        if files_with_input:
            f.write(f"### 包含input字段的文件 ({len(files_with_input)}):\n\n")
            for file in files_with_input:
                f.write(f"- ✓ `{file}`\n")
            f.write("\n")

        if files_without_input:
            f.write(f"### 不包含input字段的文件 ({len(files_without_input)}):\n\n")
            for file in files_without_input:
                f.write(f"- ✗ `{file}`\n")
            f.write("\n")

        # Order参数统计
        f.write("## 3. Config参数统计\n\n")
        f.write("### 3.1 Order参数\n\n")
        if order_stats:
            f.write("| Order值 | 文件数量 | 文件列表 |\n")
            f.write("|---------|---------|----------|\n")
            for order_val in sorted(order_stats.keys(), key=lambda x: (x is None, x)):
                files = order_stats[order_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                order_display = "null" if order_val is None else order_val
                f.write(f"| {order_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无order参数数据*\n")
        f.write("\n")

        # Need_reset参数统计
        f.write("### 3.2 Need_reset参数\n\n")
        if need_reset_stats:
            f.write("| Need_reset值 | 文件数量 | 文件列表 |\n")
            f.write("|--------------|---------|----------|\n")
            for reset_val in sorted(need_reset_stats.keys(), key=lambda x: (x is None, x)):
                files = need_reset_stats[reset_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                reset_display = "null" if reset_val is None else str(reset_val)
                f.write(f"| {reset_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无need_reset参数数据*\n")
        f.write("\n")

        # Kernel参数统计
        f.write("### 3.3 Kernel参数\n\n")
        if kernel_stats:
            f.write("| Kernel值 | 文件数量 | 文件列表 |\n")
            f.write("|----------|---------|----------|\n")
            for kernel_val in sorted(kernel_stats.keys(), key=lambda x: (x is None, x)):
                files = kernel_stats[kernel_val]
                files_str = "<br>".join([f"`{f}`" for f in files])
                kernel_display = "null" if kernel_val is None else kernel_val
                f.write(f"| {kernel_display} | {len(files)} | {files_str} |\n")
        else:
            f.write("*无kernel参数数据*\n")
        f.write("\n")

        # Total_time参数统计
        f.write("### 3.4 Total_time参数\n\n")
        if total_time_stats:
            # 基本统计信息
            f.write("#### 基本统计\n\n")
            f.write(f"- **有total_time字段的文件数**: {total_time_stats['files_with_total_time']}\n")
            f.write(f"- **无total_time字段的文件数**: {total_time_stats['files_without_total_time']}\n")
            if total_time_stats['min_value'] is not None:
                f.write(f"- **最小值**: {total_time_stats['min_value']}\n")
                f.write(f"- **最大值**: {total_time_stats['max_value']}\n")
                f.write(f"- **平均值**: {total_time_stats['avg_value']:.2f}\n")
            f.write("\n")

            # 值分布表
            f.write("#### 值分布\n\n")
            value_distribution = total_time_stats['value_distribution']
            if value_distribution:
                f.write("| Total_time值 | 文件数量 | 文件列表 |\n")
                f.write("|--------------|---------|----------|\n")
                for time_val in sorted(value_distribution.keys(), key=lambda x: (x is None, x if x is not None else 0)):
                    files = value_distribution[time_val]
                    files_str = "<br>".join([f"`{f}`" for f in files])
                    time_display = "null" if time_val is None else time_val
                    f.write(f"| {time_display} | {len(files)} | {files_str} |\n")
            f.write("\n")
        else:
            f.write("*无total_time参数数据*\n")
        f.write("\n")

        # 所有配置项统计
        f.write("### 3.5 所有配置项汇总\n\n")
        if all_config_keys:
            f.write("在所有JSON文件的config字段中，出现过的所有配置项：\n\n")
            for key in sorted(all_config_keys):
                f.write(f"- `{key}`\n")
        else:
            f.write("*无配置项数据*\n")
        f.write("\n")

        # 详细配置表
        f.write("## 4. 详细配置表\n\n")
        if config_stats:
            f.write("| 文件 | Order | Need_reset | Kernel | 其他配置项 |\n")
            f.write("|------|-------|------------|--------|------------|\n")
            for file_path in sorted(config_stats.keys()):
                config = config_stats[file_path]
                order = "null" if config['order'] is None else config['order']
                need_reset = "null" if config['need_reset'] is None else str(config['need_reset'])
                kernel = "null" if config['kernel'] is None else config['kernel']

                # 其他配置项（排除order, need_reset, kernel）
                other_keys = [k for k in config['all_keys'] if k not in ['order', 'need_reset', 'kernel']]
                other_items = ", ".join([f"`{k}`" for k in other_keys]) if other_keys else "-"

                f.write(f"| `{file_path}` | {order} | {need_reset} | {kernel} | {other_items} |\n")
        else:
            f.write("*无配置数据*\n")
        f.write("\n")

        # Edge条件分析
        f.write("## 5. Edge条件分析\n\n")
        if edge_stats:
            # 基本统计
            total_edges = sum(info['edge_count'] for info in edge_stats.values())
            files_with_edges = len(edge_stats)
            files_with_reset = sum(1 for info in edge_stats.values() if any(info['has_reset']))
            
            f.write("### 5.1 基本统计\n\n")
            f.write(f"- **包含edge字段的文件数**: {files_with_edges}\n")
            f.write(f"- **边的总数**: {total_edges}\n")

            # 收集包含reset的文件列表
            files_with_reset_list = [fp for fp, info in edge_stats.items() if any(info['has_reset'])]
            f.write(f"- **包含reset的文件数**: {files_with_reset}\n")
            if files_with_reset_list:
                for file in sorted(files_with_reset_list):
                    f.write(f"  - `{file}`\n")
            f.write("\n")
            
            # 操作符统计
            f.write("### 5.2 条件操作符统计\n\n")
            all_operators = {}
            for file_path, info in edge_stats.items():
                for op in info['operators']:
                    if op not in all_operators:
                        all_operators[op] = []
                    all_operators[op].append(file_path)
            
            if all_operators:
                f.write("| 操作符 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for op in sorted(all_operators.keys()):
                    files = all_operators[op]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{op}` | {len(files)} | {files_str} |\n")
            f.write("\n")
            
            # 变量统计
            f.write("### 5.3 条件变量统计\n\n")
            all_variables = {}
            for file_path, info in edge_stats.items():
                for var in info['variables']:
                    if var not in all_variables:
                        all_variables[var] = []
                    all_variables[var].append(file_path)
            
            if all_variables:
                f.write("| 变量名 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for var in sorted(all_variables.keys()):
                    files = all_variables[var]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{var}` | {len(files)} | {files_str} |\n")
            f.write("\n")
            
            # 条件模式统计
            f.write("### 5.4 条件模式统计\n\n")
            pattern_stats = {}
            for file_path, info in edge_stats.items():
                for condition in info['conditions']:
                    patterns = extract_condition_pattern(condition)
                    for p in patterns:
                        if p not in pattern_stats:
                            pattern_stats[p] = {'count': 0, 'files': set(), 'examples': []}
                        pattern_stats[p]['count'] += 1
                        pattern_stats[p]['files'].add(file_path)
                        if len(pattern_stats[p]['examples']) < 3:  # 保留最多3个示例
                            pattern_stats[p]['examples'].append(condition)
            
            if pattern_stats:
                f.write("| 模式类型 | 出现次数 | 文件数 | 示例 |\n")
                f.write("|----------|----------|--------|------|\n")
                pattern_descriptions = {
                    'comparison': '比较 (<=, >=, <, >)',
                    'equality': '等式 (==, !=)',
                    'compound_and': '复合条件 (and)',
                    'compound_or': '复合条件 (or)',
                    'abs_function': '绝对值函数 (abs)',
                    'var_difference': '变量差值',
                    'simple': '简单条件'
                }
                for pattern in sorted(pattern_stats.keys()):
                    stats = pattern_stats[pattern]
                    desc = pattern_descriptions.get(pattern, pattern)
                    examples_str = "<br>".join([f"`{e}`" for e in stats['examples']])
                    f.write(f"| {desc} | {stats['count']} | {len(stats['files'])} | {examples_str} |\n")
            f.write("\n")
            
            # 详细边列表
            f.write("### 5.5 详细边列表\n\n")
            f.write("| 文件 | 边数 | 方向 | 条件 | 有Reset |\n")
            f.write("|------|------|------|------|--------|\n")
            for file_path in sorted(edge_stats.keys()):
                info = edge_stats[file_path]
                for i in range(info['edge_count']):
                    direction = info['edges'][i]
                    condition = info['conditions'][i]
                    has_reset = "✓" if info['has_reset'][i] else "✗"
                    # 第一行显示文件名，后续行不显示
                    if i == 0:
                        f.write(f"| `{file_path}` | {info['edge_count']} | {direction} | `{condition}` | {has_reset} |\n")
                    else:
                        f.write(f"| | | {direction} | `{condition}` | {has_reset} |\n")
            f.write("\n")
        else:
            f.write("*无edge数据*\n\n")

        # Mode统计
        f.write("## 6. Mode统计\n\n")
        if mode_stats:
            total_modes = sum(mode_stats.values())
            f.write(f"- **包含mode字段的文件数**: {len(mode_stats)}\n")
            f.write(f"- **Mode的总数**: {total_modes}\n\n")

            # 按mode数量分组统计
            mode_count_distribution = {}
            for file_path, count in mode_stats.items():
                if count not in mode_count_distribution:
                    mode_count_distribution[count] = []
                mode_count_distribution[count].append(file_path)

            f.write("### 6.1 Mode数量分布\n\n")
            f.write("| Mode数量 | 文件数 | 文件列表 |\n")
            f.write("|---------|--------|----------|\n")
            for count in sorted(mode_count_distribution.keys()):
                files = mode_count_distribution[count]
                files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                f.write(f"| {count} | {len(files)} | {files_str} |\n")
            f.write("\n")

            f.write("### 6.2 各文件Mode详情\n\n")
            f.write("| 文件 | Mode数量 |\n")
            f.write("|------|---------|\n")
            for file_path in sorted(mode_stats.keys()):
                f.write(f"| `{file_path}` | {mode_stats[file_path]} |\n")
            f.write("\n")
        else:
            f.write("*无mode数据*\n\n")

        # Mode方程分析
        f.write("## 7. Mode方程项分析\n\n")
        if equation_stats:
            # 基本统计
            files_with_equations = len(equation_stats)
            consistent_count = sum(1 for info in equation_stats.values() if info['terms_consistent'])
            inconsistent_count = files_with_equations - consistent_count

            f.write("### 7.1 方程项一致性统计\n\n")
            f.write(f"- **包含方程的文件数**: {files_with_equations}\n")
            f.write(f"- **所有mode方程项一致的文件数**: {consistent_count}\n")
            f.write(f"- **存在mode方程项不一致的文件数**: {inconsistent_count}\n\n")

            # 方程项一致性分类
            if consistent_count > 0:
                f.write("#### 方程项一致的文件:\n\n")
                for file_path in sorted(equation_stats.keys()):
                    info = equation_stats[file_path]
                    if info['terms_consistent']:
                        terms_str = ", ".join([f"`{t}`" for t in info['all_terms']])
                        f.write(f"- ✓ `{file_path}` (共 {info['mode_count']} 个mode): {terms_str}\n")
                f.write("\n")

            if inconsistent_count > 0:
                f.write("#### 方程项不一致的文件:\n\n")
                for file_path in sorted(equation_stats.keys()):
                    info = equation_stats[file_path]
                    if not info['terms_consistent']:
                        f.write(f"- ✗ `{file_path}` (共 {info['mode_count']} 个mode)\n")
                f.write("\n")

            # 收集所有出现过的项
            f.write("### 7.2 所有方程项汇总\n\n")
            all_terms_global = set()
            for info in equation_stats.values():
                all_terms_global.update(info['all_terms'])

            if all_terms_global:
                f.write("在所有automaton的方程中，出现过的所有项（不含系数）：\n\n")
                for term in sorted(all_terms_global):
                    # 统计该项出现在多少个文件中
                    files_with_term = [fp for fp, info in equation_stats.items() if term in info['all_terms']]
                    f.write(f"- `{term}` (出现在 {len(files_with_term)} 个文件中)\n")
                f.write("\n")

            # 详细方程表
            f.write("### 7.3 详细方程列表\n\n")
            for file_path in sorted(equation_stats.keys()):
                info = equation_stats[file_path]
                consistency_icon = "✓ 一致" if info['terms_consistent'] else "✗ 不一致"
                f.write(f"#### `{file_path}` ({consistency_icon})\n\n")

                f.write("| Mode ID | 方程 | 提取的项（不含系数） |\n")
                f.write("|---------|------|----------------------|\n")
                for mode in info['modes']:
                    mode_id = mode['id']
                    equation = mode['equation']
                    terms = mode['terms']
                    terms_str = ", ".join([f"`{t}`" for t in terms])
                    f.write(f"| {mode_id} | `{equation}` | {terms_str} |\n")
                f.write("\n")

                # 如果不一致，显示差异
                if not info['terms_consistent']:
                    f.write("**差异分析**:\n\n")
                    for mode in info['modes']:
                        mode_terms = set(mode['terms'])
                        first_terms = set(info['first_mode_terms'])

                        only_in_current = mode_terms - first_terms
                        only_in_first = first_terms - mode_terms

                        if only_in_current or only_in_first:
                            f.write(f"- Mode {mode['id']}:\n")
                            if only_in_current:
                                terms_str = ", ".join([f"`{t}`" for t in sorted(only_in_current)])
                                f.write(f"  - 独有项: {terms_str}\n")
                            if only_in_first:
                                terms_str = ", ".join([f"`{t}`" for t in sorted(only_in_first)])
                                f.write(f"  - 缺少项: {terms_str}\n")
                    f.write("\n")
        else:
            f.write("*无方程数据*\n\n")

        # 变量统计
        f.write("## 8. 变量统计\n\n")
        if variable_stats:
            total_files = len(variable_stats)
            files_with_inputs = sum(1 for info in variable_stats.values() if info['input_count'] > 0)

            f.write(f"- **包含var字段的文件数**: {total_files}\n")
            f.write(f"- **包含input变量的文件数**: {files_with_inputs}\n\n")

            # 按状态变量数量分组统计
            var_count_distribution = {}
            for file_path, info in variable_stats.items():
                count = info['var_count']
                if count not in var_count_distribution:
                    var_count_distribution[count] = []
                var_count_distribution[count].append(file_path)

            f.write("### 8.1 状态变量数量分布\n\n")
            f.write("| 变量数量 | 文件数 | 文件列表 |\n")
            f.write("|---------|--------|----------|\n")
            for count in sorted(var_count_distribution.keys()):
                files = var_count_distribution[count]
                files_str = "<br>".join([f"`{f}`" for f in sorted(files)])
                f.write(f"| {count} | {len(files)} | {files_str} |\n")
            f.write("\n")

            # 各文件变量详情
            f.write("### 8.2 各文件变量详情\n\n")
            f.write("| 文件 | 状态变量数 | 状态变量 | 输入变量数 | 输入变量 | 总变量数 |\n")
            f.write("|------|-----------|---------|-----------|---------|---------|\n")
            for file_path in sorted(variable_stats.keys()):
                info = variable_stats[file_path]
                var_names = ", ".join([f"`{v}`" for v in info['var_names']]) if info['var_names'] else "-"
                input_names = ", ".join([f"`{v}`" for v in info['input_names']]) if info['input_names'] else "-"
                f.write(f"| `{file_path}` | {info['var_count']} | {var_names} | {info['input_count']} | {input_names} | {info['total_count']} |\n")
            f.write("\n")

            # 所有出现过的变量名汇总
            f.write("### 8.3 所有变量名汇总\n\n")
            all_var_names = {}
            all_input_names = {}
            for file_path, info in variable_stats.items():
                for v in info['var_names']:
                    if v not in all_var_names:
                        all_var_names[v] = []
                    all_var_names[v].append(file_path)
                for v in info['input_names']:
                    if v not in all_input_names:
                        all_input_names[v] = []
                    all_input_names[v].append(file_path)

            if all_var_names:
                f.write("#### 状态变量\n\n")
                f.write("| 变量名 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for var in sorted(all_var_names.keys()):
                    files = all_var_names[var]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{var}` | {len(files)} | {files_str} |\n")
                f.write("\n")

            if all_input_names:
                f.write("#### 输入变量\n\n")
                f.write("| 变量名 | 出现次数(文件数) | 文件列表 |\n")
                f.write("|--------|------------------|----------|\n")
                for var in sorted(all_input_names.keys()):
                    files = all_input_names[var]
                    files_str = ", ".join([f"`{f}`" for f in sorted(files)])
                    f.write(f"| `{var}` | {len(files)} | {files_str} |\n")
                f.write("\n")
        else:
            f.write("*无变量数据*\n\n")

    print(f"\n📄 分析报告已保存到: {report_path}")
    return report_path


def main():
    """主函数：统计所有JSON文件中的input字段数量和config配置"""
    # 获取automata目录的路径
    current_dir = Path(__file__).parent
    automata_dir = current_dir

    print(f"扫描目录: {automata_dir}")
    print("=" * 80)

    # 查找所有JSON文件
    json_files = find_all_json_files(automata_dir)
    json_files.sort()  # 排序便于查看

    # 统计每个文件的input字段和config配置
    total_count = 0
    files_with_input = []
    files_without_input = []
    config_stats = {}
    order_stats = {}
    need_reset_stats = {}
    kernel_stats = {}
    all_config_keys = set()
    edge_stats = {}
    mode_stats = {}  # 统计每个文件的mode数量
    equation_stats = {}  # 统计每个文件的mode方程
    variable_stats = {}  # 统计每个文件的变量数量

    for json_file in json_files:
        # 统计input字段
        count = count_input_in_json(json_file)
        total_count += count

        # 获取相对路径便于显示
        rel_path = os.path.relpath(json_file, automata_dir)

        if count > 0:
            files_with_input.append(rel_path)
            print(f"✓ {rel_path}")
        else:
            files_without_input.append(rel_path)
            print(f"✗ {rel_path}")

        # 分析config配置
        config_info = analyze_config_in_json(json_file)
        if config_info:
            config_stats[rel_path] = config_info

            # 统计order值
            order_val = config_info['order']
            if order_val not in order_stats:
                order_stats[order_val] = []
            order_stats[order_val].append(rel_path)

            # 统计need_reset值
            need_reset_val = config_info['need_reset']
            if need_reset_val not in need_reset_stats:
                need_reset_stats[need_reset_val] = []
            need_reset_stats[need_reset_val].append(rel_path)

            # 统计kernel值
            kernel_val = config_info['kernel']
            if kernel_val not in kernel_stats:
                kernel_stats[kernel_val] = []
            kernel_stats[kernel_val].append(rel_path)

            # 收集所有配置键
            all_config_keys.update(config_info['all_keys'])

        # 分析edge条件
        edge_info = analyze_edges_in_json(json_file)
        if edge_info:
            edge_stats[rel_path] = edge_info

        # 统计mode数量
        mode_count = count_mode_in_json(json_file)
        if mode_count > 0:
            mode_stats[rel_path] = mode_count

        # 分析mode方程
        equation_info = analyze_mode_equations(json_file)
        if equation_info:
            equation_stats[rel_path] = equation_info

        # 分析变量数量
        var_info = analyze_variables_in_json(json_file)
        if var_info:
            variable_stats[rel_path] = var_info

    # 打印统计结果
    print("=" * 80)
    print(f"\n统计结果:")
    print(f"  总JSON文件数: {len(json_files)}")
    print(f"  包含input字段的文件数: {len(files_with_input)}")
    print(f"  不包含input字段的文件数: {len(files_without_input)}")
    print(f"  包含config字段的文件数: {len(config_stats)}")
    # 收集total_time统计
    total_time_stats = get_total_time_stats(config_stats)

    print(f"\nConfig参数统计:")
    print(f"  Order参数分布: {dict((k, len(v)) for k, v in order_stats.items())}")
    print(f"  Need_reset参数分布: {dict((k, len(v)) for k, v in need_reset_stats.items())}")
    print(f"  Kernel参数分布: {dict((k, len(v)) for k, v in kernel_stats.items())}")
    print(f"  Total_time参数分布: {dict((k, len(v)) for k, v in total_time_stats['value_distribution'].items())}")
    if total_time_stats['min_value'] is not None:
        print(f"  Total_time统计: 最小={total_time_stats['min_value']}, 最大={total_time_stats['max_value']}, 平均={total_time_stats['avg_value']:.2f}")
    print(f"  所有配置项: {sorted(all_config_keys)}")

    # 打印Edge统计结果
    print(f"\nEdge条件统计:")
    print(f"  包含edge字段的文件数: {len(edge_stats)}")
    total_edges = sum(info['edge_count'] for info in edge_stats.values())
    print(f"  边的总数: {total_edges}")
    
    # 统计所有操作符
    all_operators = set()
    all_variables = set()
    for info in edge_stats.values():
        all_operators.update(info['operators'])
        all_variables.update(info['variables'])
    print(f"  使用的操作符: {sorted(all_operators)}")
    print(f"  使用的变量: {sorted(all_variables)}")

    # 打印Mode统计结果
    print(f"\nMode统计:")
    print(f"  包含mode字段的文件数: {len(mode_stats)}")
    total_modes = sum(mode_stats.values())
    print(f"  Mode的总数: {total_modes}")
    print(f"  各文件Mode数量:")
    for file_path, count in sorted(mode_stats.items()):
        print(f"    {file_path}: {count}")

    # 打印Mode方程分析结果
    print(f"\nMode方程分析:")
    print(f"  包含方程的文件数: {len(equation_stats)}")
    consistent_count = sum(1 for info in equation_stats.values() if info['terms_consistent'])
    print(f"  方程项一致的文件数: {consistent_count}")
    print(f"  方程项不一致的文件数: {len(equation_stats) - consistent_count}")

    # 打印变量统计结果
    print(f"\n变量统计:")
    print(f"  包含var字段的文件数: {len(variable_stats)}")
    for file_path, info in sorted(variable_stats.items()):
        input_str = f", 输入: {', '.join(info['input_names'])}" if info['input_names'] else ""
        print(f"    {file_path}: {info['var_count']}个状态变量 ({', '.join(info['var_names'])}){input_str}")

    # 生成Markdown报告
    generate_markdown_report(automata_dir, json_files, files_with_input, files_without_input,
                            config_stats, order_stats, need_reset_stats, kernel_stats,
                            all_config_keys, edge_stats, mode_stats, equation_stats,
                            total_time_stats, variable_stats)


if __name__ == "__main__":
    main()

