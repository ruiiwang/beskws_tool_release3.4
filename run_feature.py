import os
import random
import wave
from tqdm import tqdm
import numpy as np
import soundfile as sf
import subprocess
from multiprocessing import Pool
import torch

from src.feature.featurizer import extract_feature_


def read_list(file):
    with open(file, 'r') as f:
        return [line.strip().split() for line in f]


def get_wav_info(wav_file):
    try:
        with wave.open(wav_file, 'rb') as wav:
            sample_rate = wav.getframerate()  # 获取采样率
            num_channels = wav.getnchannels()  # 通道数
            num_frames = wav.getnframes()  # 获取总帧数
            duration = num_frames / float(sample_rate)  # 计算时长（秒）
            sample_width = wav.getsampwidth()
            # 位深 = 字节数 * 8
            bit_depth = sample_width * 8
        return sample_rate, duration, num_channels, bit_depth
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return None, None, None, None


def filter(datas, sample_rate=16000, min_dur=0.2, max_dur=2.5):
    """
    过滤文件列表，返回时长在 [min_dur, max_dur] 范围内的 WAV 文件。

    :param datas: 文件路径list 或 [文件路径，标签] 的list
    :param min_dur: 最小时长（秒）
    :param max_dur: 最大时长（秒）
    :return: 过滤后的数据
    """
    valid_datas = []

    for x in tqdm(datas, desc="filter wav"):
        if isinstance(x, list):
            file = x[0]
        else:
            file = x

        sr, duration, num_channels, bit_depth = get_wav_info(file)
        if (sr == sample_rate and num_channels == 1 and bit_depth == 16
                and min_dur <= duration <= max_dur):
            valid_datas.append(x)

    return valid_datas


def cal_feature(data):
    if isinstance(data, list):
        file_path, label = data
    else:
        file_path = data
        label = ''

    nn_input_ms = (nn_input_frames - 1) * FbankArgs['frame_shift'] \
                  + FbankArgs['frame_length']
    sample_rate = FbankArgs['sample_frequency']
    nn_input_len = int(nn_input_ms * sample_rate / 1000)  # 输入的长度

    # Load the audio file
    # audio, sr = torchaudio.load(file_path)
    # audio_len = audio.shape[-1]
    audio, sr = sf.read(file_path, dtype='float32')
    audio_len = len(audio)
    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, but got {sr}")

    if audio_len == 0:
        return None

    # 将数据插入都0张量中，实现了padding
    input_data = np.zeros((nn_input_len), dtype='float32')
    # input_data = torch.zeros((nn_input_len), dtype=torch.float32)

    if audio_len > nn_input_len:
        audio_len = nn_input_len

    if random.random() > 0.5:
        input_data[-audio_len:] = audio[:audio_len]
    else:
        input_data[:audio_len] = audio[:audio_len]

    # 转换为 PyTorch Tensor 并传输到目标设备
    input_tensor = torch.from_numpy(input_data).unsqueeze(0)
    # Extract feature
    feat = extract_feature_(input_tensor, FbankArgs)

    return {"data": feat, "label": label}


def cal_feature_C(data):
    if isinstance(data, list):
        file_path, label = data
    else:
        file_path = data
        label = ''

    padding_head = 0 if random.random() > 0.5 else 1

    command = ['Keyword/mfcc_test', file_path, str(padding_head)]
    result = subprocess.run(command, capture_output=True, text=True)

    # 获取程序的标准输出
    output = result.stdout

    # 解析输出（假设输出是一个空格分隔的数组）
    feat = np.array([np.float32(val) for val in output.split()])
    feat = feat.reshape(-1, FbankArgs['num_mel_bins'])

    if feat.shape[0] != nn_input_frames:
        print(file_path, feat.shape)
        return None
    else:
        return {"data": feat, "label": label}


def run_process(datas, out_npy, device, process_num=1):
    results = []
    if process_num == 1:
        for data in tqdm(datas, total=len(datas), desc="Processing"):
            result = cal_feature(data)
            results.append(result)
    else:
        with Pool(processes=process_num) as pool:
            results = list(tqdm(
                pool.imap(cal_feature_C, datas),
                total=len(datas),
                desc="Processing cal_feature_C"))
    results = [x for x in results if x is not None]
    np.save(out_npy, results)


# 计算Fbank的参数
FbankArgs = {
    'sample_frequency': 16000,
    'num_mel_bins': 40,
    'frame_length': 32.0,
    'frame_shift': 20.0,
    'high_freq': 0.0,
    'low_freq': 20.0,
    'preemphasis_coefficient': 0.0,
    'raw_energy': False,
    'remove_dc_offset': False,
    'use_power': True,  # False,
    'window_type': 'hanning'
}
nn_input_frames = 100

if __name__ == '__main__':
    # 定义设备
    device = torch.device("cpu")

    # 参数设定
    process_num = 20
    list_file = "Keyword/datas/kws_datas/training_datas.txt"
    out_dir = "Keyword/datas/kws_datas"
    feat_npy = f"{out_dir}/train.npy"

    os.makedirs(out_dir, exist_ok=True)
    print(f"process_num {process_num}")

    datas = read_list(list_file)
    datas = filter(datas)

    run_process(datas, feat_npy, device, process_num)
    datas = np.load(feat_npy, allow_pickle=True).tolist()
    for data in datas:
        if data['data'].shape != (nn_input_frames, FbankArgs['num_mel_bins']):
            print(data['data'].shape)
