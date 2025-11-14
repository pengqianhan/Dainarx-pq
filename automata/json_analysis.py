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
                'all_keys': list(config.keys())
            }
            return result
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
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
                            all_config_keys):
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

        # 所有配置项统计
        f.write("### 3.4 所有配置项汇总\n\n")
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

    # 打印统计结果
    print("=" * 80)
    print(f"\n统计结果:")
    print(f"  总JSON文件数: {len(json_files)}")
    print(f"  包含input字段的文件数: {len(files_with_input)}")
    print(f"  不包含input字段的文件数: {len(files_without_input)}")
    print(f"  包含config字段的文件数: {len(config_stats)}")
    print(f"\nConfig参数统计:")
    print(f"  Order参数分布: {dict((k, len(v)) for k, v in order_stats.items())}")
    print(f"  Need_reset参数分布: {dict((k, len(v)) for k, v in need_reset_stats.items())}")
    print(f"  Kernel参数分布: {dict((k, len(v)) for k, v in kernel_stats.items())}")
    print(f"  所有配置项: {sorted(all_config_keys)}")

    # 生成Markdown报告
    generate_markdown_report(automata_dir, json_files, files_with_input, files_without_input,
                            config_stats, order_stats, need_reset_stats, kernel_stats,
                            all_config_keys)


if __name__ == "__main__":
    main()

