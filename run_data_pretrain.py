#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import soundfile as sf
from tqdm import tqdm

def check_wav_file(wav_file):
    """检查WAV文件是否可以正常读取"""
    try:
        if not os.path.exists(wav_file):
            print(f"文件不存在: {wav_file}")
            return False
        sf.read(wav_file)
        return True
    except Exception as e:
        print(f"文件 {wav_file} 无法读取: {e}")
        return False

def filter_dataset(input_file, output_file=None):
    """过滤数据集中不可读取的WAV文件"""
    if output_file is None:
        # 如果没有指定输出文件，生成默认名称
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"
    
    # 读取原始数据列表
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"原始数据行数: {len(lines)}")
    
    # 过滤掉不可读取的文件
    valid_lines = []
    invalid_count = 0
    
    for line in tqdm(lines, desc="检查文件"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 1:
            print(f"格式不正确的行: {line}")
            invalid_count += 1
            continue
            
        wav_file = parts[0]
        
        if check_wav_file(wav_file):
            valid_lines.append(line + '\n')
        else:
            invalid_count += 1
    
    # 保存有效的数据列表
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    print(f"原始数据行数: {len(lines)}")
    print(f"无效文件数量: {invalid_count}")
    print(f"过滤后行数: {len(valid_lines)}")
    print(f"有效数据已保存至: {output_file}")

if __name__ == "__main__":
    # 在这里直接设置输入和输出文件路径
    input_file = "datas/kws_datas/training_datas_origin.txt"  # 修改为您的输入文件路径
    output_file = "datas/kws_datas/training_datas_origin.txt"  # 修改为您的输出文件路径
    
    filter_dataset(input_file, output_file)
