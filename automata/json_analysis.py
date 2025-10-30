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


def main():
    """主函数：统计所有JSON文件中的input字段数量"""
    # 获取automata目录的路径
    current_dir = Path(__file__).parent
    automata_dir = current_dir
    
    print(f"扫描目录: {automata_dir}")
    print("=" * 80)
    
    # 查找所有JSON文件
    json_files = find_all_json_files(automata_dir)
    json_files.sort()  # 排序便于查看
    
    # 统计每个文件的input字段
    total_count = 0
    files_with_input = []
    files_without_input = []
    
    for json_file in json_files:
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
    
    # 打印统计结果
    print("=" * 80)
    print(f"\n统计结果:")
    print(f"  总JSON文件数: {len(json_files)}")
    print(f"  包含input字段的文件数: {len(files_with_input)}")
    print(f"  不包含input字段的文件数: {len(files_without_input)}")
    print(f"  input字段总数: {total_count}")
    
    # 详细列表
    if files_with_input:
        print(f"\n包含input字段的文件 ({len(files_with_input)}):")
        for file in files_with_input:
            print(f"  - {file}")
    
    if files_without_input:
        print(f"\n不包含input字段的文件 ({len(files_without_input)}):")
        for file in files_without_input:
            print(f"  - {file}")


if __name__ == "__main__":
    main()

