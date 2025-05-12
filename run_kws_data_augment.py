import os
import logging
from src.utils.utils import dict_to_object
from src.data_process.data_augment import data_aug

# 数据增强参数
train_aug_conf = {
    # 做音频增强的概率,整数， 0:不做增强
    'aug_ratio': 8,
    # 重采样增强
    'use_resample': False,
    # 各数据增强的概率：time_shift, volume, pitch, tempo, speed, noise and reverberation ,
    'aug_prob': [1.0, 0.6, 0.9, 0.0, 0.9, 0.9, 0.3],
    # 时移范围
    'shift_ms': [-100, 100],
    # 音量变化范围
    'volumes': [1.0, 1.4],
    # pitch变化范围
    'pitchs': [-2, 2],
    # tempo变化范围
    'tempos': [0.9, 1.1],
    # speed变换的可选数值
    'speeds': [1.0, 0.9, 1.1],
    # 加噪信噪比范围
    'snrs': [-12, 18],
    # 噪声增强的噪声文件夹
    'noise_dir': 'datas/noise',
    # 混响增强的混响文件夹
    'rir_dir': 'datas/rirs_noises',
    # 噪声数据按类型选择的概率/比例, 需要和为1.0
    'noise_percentage': {
        'bathroom': 0.12,
        'beach': 0.12,
        'cafe': 0.01,
        'car': 0.01,
        'fire': 0.03,
        'kitchen': 0.12,
        'market': 0.12,
        'playground': 0.11,
        'rain': 0.11,
        'transport': 0.14,
        'voice': 0.1,
        'white': 0.01,
    },
}

if __name__ == '__main__':
    # 超参数设置
    process_num = 20  # 启用多线程数
    sample_rate = 16000  # wav音频数据的采样率

    # 数据路径和增强配置参数设置
    data_dirs = [
        "datas/edgetts_generated",
        "datas/human_modified_153",
        "datas/orpheus_generated",
        "datas/unknown_data",
    ]
    out_dir = "datas/train_data_augment"
    aug_conf = train_aug_conf

    # 创建输出文件夹
    os.makedirs(out_dir, exist_ok=True)

    # 设置日志级别、输出格式和输出内容等
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    aug_conf = dict_to_object(aug_conf)
    for data_dir in data_dirs:
        data_aug(data_dir=data_dir,
                 out_dir=out_dir,
                 sample_rate=sample_rate,
                 process_num=process_num,
                 aug_conf=aug_conf)
