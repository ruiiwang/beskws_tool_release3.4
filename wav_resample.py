import os
import librosa
import soundfile as sf
import numpy as np


def resample_wav_file(input_path, target_sample_rate=16000):
    """
    重新采样WAV文件到指定的采样率，并删除旧文件

    参数:
    input_path (str): 输入WAV文件路径
    target_sample_rate (int, 可选): 目标采样率，默认为16000 Hz

    返回:
    str: 输出文件的完整路径
    """
    try:
        # 读取原始音频文件
        y, orig_sr = librosa.load(input_path, sr=None)

        # 重新采样
        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sample_rate)

        # 保存重采样后的音频文件（覆盖原文件）
        sf.write(input_path, y_resampled, target_sample_rate)

        print(f"重采样完成：{input_path}")
        print(f"原始采样率：{orig_sr} Hz, 目标采样率：{target_sample_rate} Hz")

        return input_path

    except Exception as e:
        print(f"重采样时发生错误: {e}")
        return None


def batch_resample_wav_files(input_folder, target_sample_rate=16000):
    """
    批量重采样文件夹中的WAV文件，并删除旧文件

    参数:
    input_folder (str): 输入WAV文件所在文件夹
    target_sample_rate (int, 可选): 目标采样率，默认为16000 Hz

    返回:
    list: 重采样后的文件路径列表
    """
    resampled_files = []

    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)

            # 重采样单个文件
            resampled_file = resample_wav_file(input_path, target_sample_rate=target_sample_rate)

            if resampled_file:
                resampled_files.append(resampled_file)

    return resampled_files


def recursive_batch_resample_wav_files(root_folder, target_sample_rate=16000):
    """
    递归重采样文件夹及其所有子文件夹中的WAV文件，并删除旧文件

    参数:
    root_folder (str): 根文件夹路径
    target_sample_rate (int, 可选): 目标采样率，默认为16000 Hz

    返回:
    list: 重采样后的文件路径列表
    """
    all_resampled_files = []

    # 遍历根文件夹中的所有项目
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                input_path = os.path.join(root, file)

                # 重采样单个文件
                resampled_file = resample_wav_file(input_path, target_sample_rate=target_sample_rate)

                if resampled_file:
                    all_resampled_files.append(resampled_file)

    return all_resampled_files


# 使用示例
if __name__ == '__main__':
    # 单文件夹批量重采样
    input_folder = 'D:/project/LooktechVoice/results/SPK004'
    batch_resample_wav_files(input_folder, target_sample_rate=16000)

    # 递归重采样（包括所有子文件夹）
    root_folder = 'D:/project/LooktechVoice/VoiceGeneration/edgetts_generated_new'
    recursive_batch_resample_wav_files(root_folder, target_sample_rate=16000)
