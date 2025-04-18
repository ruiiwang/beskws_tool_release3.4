#!/usr/bin/env python
# -*- coding: utf-8 -*-

def cat_files(input_files, output_file):
    """
    将多个输入文件的内容合并到一个输出文件中

    参数:
        input_files: 输入文件路径列表
        output_file: 输出文件路径
    """
    # 打开输出文件用于写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历每个输入文件
        for input_file in input_files:
            try:
                # 打开输入文件用于读取
                with open(input_file, 'r', encoding='utf-8') as infile:
                    # 读取输入文件内容并写入输出文件
                    outfile.write(infile.read())
            except FileNotFoundError:
                print(f"错误: 找不到文件 '{input_file}'")
            except Exception as e:
                print(f"处理文件 '{input_file}' 时出错: {str(e)}")


if __name__ == "__main__":
    # 硬编码的文件路径列表（可以添加任意数量的输入文件）
    input_files = [
        "datas/kws_datas/training_datas_orpheus.txt",
        "datas/kws_datas/training_datas_edgetts.txt",
        "datas/kws_datas/training_datas.txt",
        "datas/kws_datas/training_datas_unknown.txt"
        # 可以继续添加更多文件...
    ]

    # 输出文件路径
    output_file = "datas/kws_datas/training_datas_origin.txt"

    # 调用函数合并文件
    cat_files(input_files, output_file)

    # 构建输出信息
    input_files_str = "、".join(input_files)
    print(f"已成功将 {input_files_str} 的内容合并到 {output_file}")
