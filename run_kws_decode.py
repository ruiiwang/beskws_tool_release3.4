import os
import logging
import torch
import numpy as np
import soundfile as sf
from src.model.tcn import TCN2
from src.feature.featurizer import extract_feature


def search_wavs(root_dir):
    wav_files = []
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名是否为.wav
            if file.endswith(".wav"):
                # 构建完整的文件路径并添加到列表
                wav_files.append(os.path.join(root, file))

    return wav_files


def load_model(model_dir, num_class, device, input_size=40):
    """ 加载模型并返回模型实例 """
    # 检查模型目录是否存在
    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory '{model_dir}' does not exist.")

    # 确认模型文件是否存在
    model_file = os.path.join(model_dir, 'model.pth')
    if not os.path.isfile(model_file):
        raise ValueError(f"Model file '{model_file}' does not exist.")

    # 创建模型实例
    model = TCN2(in_channels=input_size, num_class=num_class)
    # 加载模型状态字典
    state_dict = torch.load(model_file, map_location=device)
    # 处理状态字典中的键名，移除'_orig_mod.'前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # 移除前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 加载模型参数
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)

    logging.debug(f"Model loaded successfully from '{model_file}' on device '{device}'")

    return model


def decode(model, files, device, sample_rate):
    """ 逐个样本解码
    """
    result = {}

    nn_input_len = int(2012 * sample_rate / 1000)  # 输入的长度

    with torch.no_grad():
        for file in files:

            audio, _ = sf.read(file, dtype='float32')
            audio_len = len(audio)

            input_data = np.zeros((1, nn_input_len), dtype='float32')
            # 将数据插入都0张量中，实现了padding
            if audio_len > nn_input_len:
                audio_len = nn_input_len
            input_data[0, -audio_len:] = audio[:audio_len]

            # 转换为 PyTorch Tensor 并传输到目标设备
            input_tensor = torch.from_numpy(input_data).to(device)

            feature = extract_feature(input_tensor, FbankArgs, device)

            output = model(feature)
            output = torch.nn.functional.softmax(output, dim=-1)

            # 转换为 numpy 数组并获取最大值索引
            output_numpy = output.cpu().numpy()[0]
            index = np.argmax(output_numpy)

            logging.info(f"{file}, idx: {index}, score {output_numpy[index]:.3f}")

            result[file] = [index, output_numpy[index]]

    return result


def slide_window_decode(model, files, device, sample_rate, frame_shift):
    """仿真流式解码
    """
    frame_shift_len = int(frame_shift * sample_rate / 1000)  # 帧长
    nn_input_len = int(2012 * sample_rate / 1000)  # 输入的长度
    padding_len = int(1000 * sample_rate / 1000)

    UNKNOWN_WORD_INDEX = 0  # 未知词索引
    threshold = 0.9  # 阈值

    # 关闭梯度计算，提高性能
    with torch.no_grad():
        for file in files:
            # 读取音频文件
            audio, _ = sf.read(file, dtype='float32')
            audio_len = len(audio)

            # 如果音频长度小于 nn_input_len，填充零
            if audio_len < nn_input_len:
                input_tensor = torch.zeros(nn_input_len, dtype=torch.float32)
                input_tensor[:audio_len] = torch.tensor(audio)
                audio_len = nn_input_len
            else:
                input_tensor = torch.tensor(audio, dtype=torch.float32)

            input_tensor = torch.nn.functional.pad(input_tensor, (padding_len, padding_len), mode='constant', value=0)
            audio_len += padding_len * 2

            # 将输入转换为 batch 形式
            input_tensor = input_tensor[None, :].to(device)

            result_str = f"{file}\n"  # 初始化结果字符串
            i = 0

            # 使用滑动窗口进行解码
            while i + nn_input_len <= audio_len:
                # 获取当前窗口的数据
                input_window = input_tensor[:, i:i + nn_input_len]

                # 提取特征
                feature = extract_feature(input_window, FbankArgs, device)

                # 通过模型进行推理
                output = model(feature)
                output = torch.nn.functional.softmax(output, dim=-1)

                # 将输出转换为 NumPy 数组以便处理
                output_numpy = output.cpu().numpy()[0]

                # 获取最大概率的索引及其对应的得分
                index = np.argmax(output_numpy, axis=0)
                score = output_numpy[index]

                # 如果索引不是未知词且得分超过阈值，则记录结果
                if index != UNKNOWN_WORD_INDEX and score > threshold:
                    result_str += f"time: {(i + nn_input_len) / sample_rate:.2f}s, idx: {index}, score: {score:.3f}\n"
                    i += nn_input_len  # 处理一个完整窗口
                else:
                    i += frame_shift_len  # 否则滑动窗口向前移动1帧

            # 输出当前文件的解码结果
            logging.info(result_str)


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

if __name__ == '__main__':

    # 参数设置
    num_class = 11  # 根据实际唤醒词个数+1去设定（ +1，是还有一个_unknown_类）

    # 设置日志级别、输出格式和输出内容等
    logging.basicConfig(  # filename='decode_log.txt',
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    feat_dim = FbankArgs['num_mel_bins']
    sample_rate = FbankArgs['sample_frequency']
    frame_shift = FbankArgs['frame_shift']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = './models/TCN2/last_model'
    model = load_model(model_dir, num_class, device, feat_dim)
    model.eval()

    # 测试数据
    # wav_files = ["./datas/mini_speech_commands/down/0a9f9af7_nohash_0.wav",
    #              "./datas/mini_speech_commands/go/0a9f9af7_nohash_0.wav",
    #              "./datas/mini_speech_commands/left/0b09edd3_nohash_0.wav"]
    # decode(model, wav_files, device, sample_rate)
    # slide_window_decode(model, wav_files, device, sample_rate, frame_shift)

    data_dirs = ["./datas/train_data_resampled/SPK099_resampled",
                 "./datas/train_data_resampled/SPK100_resampled"]
    for data_dir in data_dirs:
        wav_files = search_wavs(data_dir)
        result = decode(model, wav_files, device, sample_rate)
        # slide_window_decode(model, wav_files, device, sample_rate, frame_shift)
