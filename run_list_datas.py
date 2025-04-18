import os
import random

"""
遍历唤醒词/命令词语音数据路径，列出wav文件与标签对
需要文件名里包含关键词字符串
"""


def list_keywords(data_dirs, out_dir, filename, words_dict):
    """
    遍历多个数据目录，生成包含 wav 文件路径和对应标签的文件。
    """
    os.makedirs(out_dir, exist_ok=True)
    # 创建输出文件路径
    out_file = os.path.join(out_dir, filename)
    with open(out_file, 'w') as fw:
        for data_dir in data_dirs:
            # 遍历当前目录及其子目录
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if not file.endswith('.wav'):
                        continue  # 仅处理 .wav 文件
                    label = UNKNOWN_WORD
                    # 查找文件名中的标签
                    for word_, label_ in words_dict.items():
                        if word_.lower() in file.lower():
                            label = label_
                            break
                    file_path = os.path.join(root, file)
                    fw.write(f'{file_path}\t{label}\n')


# 列出唤醒词/命令词数据 和 负样本（unknown）数据
# 用户需要根据自己的数据格式编译相应的函数
def list_examples(out_dir, keywords_dict):
    data_dirs = [
        f'datas/human_modified_153/SPK{str(i).zfill(3)}_resampled'
        for i in range(1, 154)
    ]
    # data_dirs = [['./datas/train_data/yes', keywords_dict['yes']],
    #             ['./datas/train_data/no', keywords_dict['no']],
    #             ['./datas/train_data/up', keywords_dict['up']],
    #             ['./datas/train_data/down', keywords_dict['down']],
    #             ['./datas/train_data/right', keywords_dict['right']],
    #             ['./datas/train_data/left', keywords_dict['left']],
    #             ["./datas/train_data/go",  keywords_dict[UNKNOWN_WORD]],
    #             ["./datas/train_data/stop",  keywords_dict[UNKNOWN_WORD]],
    #             ]
    # 从路径里寻找wav文件，并给出标签
    datas = []
    for data_dir in data_dirs:
        # for data_dir, label in data_dirs:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    # 例如: SPK001_AUS_Sydney_male_40_HeyMemo_-15.4dB_1.7wps_ts-17_v1.35_p2_sp1.1.wav
                    parts = file.split('_')
                    if len(parts) >= 6:
                        keyword = parts[5]  # 提取关键词
                        label = keywords_dict.get(keyword, keywords_dict[UNKNOWN_WORD])
                        file_path = os.path.join(root, file)
                        datas.append([f"{root}/{file}", label])

    random.shuffle(datas)  # 随机打乱
    # 将wav文件路径及其标签对写入txt文件里
    with open(f'{out_dir}/training_datas_human.txt', 'w') as fw:
        for data in datas:
            file_path, label = data
            fw.write(f'{file_path}\t{label}\n')

def other_examples(out_dir, keywords_dict):
    edgetts_data_dir = 'datas/edgetts_generated'
    datas = []
    for root, dirs, files in os.walk(edgetts_data_dir):
        for file in files:
            if file.endswith('.wav'):
                # 从文件名中提取关键词
                # 例如: AUS_Sydney_Female_25_HeyMemo_var1.wav
                parts = file.split('_')
                if len(parts) >= 5:
                    keyword = parts[4]  # 提取关键词
                    label = keywords_dict.get(keyword, keywords_dict[UNKNOWN_WORD])
                    file_path = os.path.join(root, file)
                    datas.append([file_path, label])

    random.shuffle(datas)  # 随机打乱
    # 将wav文件路径及其标签对写入txt文件里
    with open(f'{out_dir}/training_datas_edgetts.txt', 'w') as fw:
        for data in datas:
            file_path, label = data
            file_path = file_path.replace('\\', '/')
            fw.write(f'{file_path}\t{label}\n')


    orpheus_data_dir = 'datas/orpheus_generated'
    datas2 = []
    for root, dirs, files in os.walk(orpheus_data_dir):
        for file in files:
            if file.endswith('.wav'):
                # 从文件名中提取关键词
                # 例如: AUS_Sydney_Female_25_HeyMemo_var1.wav
                parts = file.split('_')
                if len(parts) >= 5:
                    keyword = parts[4]  # 提取关键词
                    label = keywords_dict.get(keyword, keywords_dict[UNKNOWN_WORD])
                    file_path = os.path.join(root, file)
                    datas2.append([file_path, label])

    random.shuffle(datas2)  # 随机打乱
    # 将wav文件路径及其标签对写入txt文件里
    with open(f'{out_dir}/training_datas_orpheus.txt', 'w') as fw:
        for data in datas2:
            file_path, label = data
            file_path = file_path.replace('\\', '/')
            fw.write(f'{file_path}\t{label}\n')
    
    unknown_data_dir = 'datas/unknown_data'
    datas3 = []
    for root, dirs, files in os.walk(unknown_data_dir):
        for file in files:
            if file.endswith('.wav'):
                label = keywords_dict[UNKNOWN_WORD]
                file_path = os.path.join(root, file)
                datas3.append([file_path, label])

    random.shuffle(datas3)  # 随机打乱
    # 将wav文件路径及其标签对写入txt文件里
    with open(f'{out_dir}/training_datas_unknown.txt', 'w') as fw:
        for data in datas3:
            file_path, label = data
            file_path = file_path.replace('\\', '/')
            fw.write(f'{file_path}\t{label}\n')


if __name__ == '__main__':
    out_dir = 'datas/kws_datas'
    os.makedirs(out_dir, exist_ok=True)

    UNKNOWN_WORD = '_unknown_'

    # 关键词(唤醒词/命令词）和 标签 的映射关系
    keywords_dict = {
        UNKNOWN_WORD: 0,
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

    # 列出关键词及其标签对
    list_examples(out_dir, keywords_dict)
    other_examples(out_dir, keywords_dict)
