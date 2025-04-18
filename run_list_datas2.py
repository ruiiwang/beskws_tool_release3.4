import os
import random

def generate_keyword_label_mapping(input_dir, output_file):
    """
    从无规律文件名中提取关键词并生成标签映射文件
    
    参数:
        input_dir: 包含.wav文件的输入目录
        output_file: 输出文件路径，格式为"文件路径\t标签"
    """
    # 关键词到标签的映射字典
    keywords_dict = {
        'Unknown': 0,
        'HeyMemo': 1,
        'Next': 2,
        'Pause': 3,
        'Play': 4,
        'StopRecording': 5,
        'TakeAPicture': 6,
        'TakeAVideo': 7,
        'VolumeDown': 8,
        'VolumeUp': 9,
        'LookAnd': 10,
    }

    def extract_keyword(filename):
        """从文件名中提取关键词"""
        filename_lower = filename.lower()
        for keyword in keywords_dict:
            if keyword.lower() in filename_lower and keyword != 'Unknown':
                return keyword
        return 'Unknown'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 收集所有文件路径和标签
    file_label_pairs = []
    
    # 遍历输入目录及其子目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                # 获取完整文件路径
                file_path = os.path.join(root, file)
                # 提取关键词和标签
                keyword = extract_keyword(file)
                label = keywords_dict[keyword]
                # 添加到列表中
                file_label_pairs.append((file_path, label))
    
    # 随机打乱顺序
    random.shuffle(file_label_pairs)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path, label in file_label_pairs:
            f_out.write(f"{file_path}\t{label}\n")

    print(f"标签映射文件已生成: {output_file}")
    print(f"共处理了 {len(list(os.walk(input_dir)))} 个目录中的文件")
    print(f"共处理了 {len(file_label_pairs)} 个文件")


# 使用示例
if __name__ == '__main__':
    input_directory = 'datas/train_data_augment'
    output_path = 'datas/kws_datas/training_datas_augment.txt'
    generate_keyword_label_mapping(input_directory, output_path)
