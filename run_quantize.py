#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import numpy as np
import torch
import logging

# from src.parse.parse_float_model import parse_float,run_float
from src.quantization.parse.parse_quant_model import parse_quant, run_quant
from src.quantization.writer.write_model_h import write_model_h
from src.quantization.quantize_model import quantize
from src.quantization.utils.kws_utils import fuse_modules, load_model, calibrate_dataset


def find_wav_files(directory, extns='.wav'):
    """
    Recursively find all .wav files in the given directory and its subdirectories.
    """
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extns):
                wav_files.append(os.path.join(root, file))
    return wav_files


def random_dataset(input_dims, sample_num):
    for _ in range(sample_num):
        yield torch.rand(input_dims)


def quantize_process(model, q_model_path, wav_files=None, sample_num=0, device=None):
    # 模型量化 #
    # 标定数据
    if wav_files is None or len(wav_files) == 0:
        dataset = random_dataset(input_dims, sample_num)  # 随机生成数据
    else:
        dataset = calibrate_dataset(wav_files, sample_num)

    # 模型输入shape
    input_dims = tuple(next(dataset).shape)

    # 量化
    if os.path.exists(q_model_path):
        model_quantized = quantize(model, fuse_modules, None)
        model_state_dict = torch.load(q_model_path, map_location=device)
        model_quantized.load_state_dict(model_state_dict)
    else:
        model_quantized = quantize(model, fuse_modules, dataset)
        # 保存量化后的模型
        torch.save(model_quantized.state_dict(), q_model_path)
    # print("model_quantized", model_quantized)

    # 解析量化模型参数
    q_layers = parse_quant(model_quantized, input_dims)

    return q_layers


def model_inference(model, q_layers, input_tensor):
    # 量化模型推理
    quant_out = run_quant(q_layers, input_tensor, False)
    # 浮点模型推理
    float_out = model(input_tensor)

    print(f"float_out : {float_out.detach().numpy()}")
    print(f"quant_out : {quant_out.detach().numpy()}")

    """
    # 解析浮点模型
    f_layers = parse_float(model, input_dims)

    # 依次运行模型每一层
    f_layers_out = run_float(f_layers, input_tensor)

    # 比较结果是否一致  
    if not torch.equal(float_out, f_layers_out):
        print(f"浮点model(x) : {float_out.detach().numpy()}")
        print(f"浮点 F.x函数 : {f_layers_out.detach().numpy()}")
    """

    # layerParams(q_layers, prefix)


def main(model_dir, wavs_dir, model_type, kws_class_num, sample_num, model_h_file, use_beco, device='cpu'):
    # wav文件列表
    wav_files = find_wav_files(wavs_dir)

    # 加载模型
    model_path = os.path.join(model_dir, "model.pth")
    if model_type == 'kws':
        model = load_model(model_path, kws_class_num, device)
    else:
        model = load_model(model_path, device)

    # 量化处理
    q_model_path = os.path.join(model_dir, f"model_quant.pth")
    q_layers = quantize_process(model, q_model_path, wav_files, sample_num, device)

    # 将量化后的模型参数写入model.h
    write_model_h(q_layers, model_h_file, model_type, kws_class_num, use_beco=use_beco)

    # 推理测试
    wav_files = [
        "./datas/edgetts_generated/AUS_Sydney_Female_25_Fast/AUS_Sydney_Female_25_HeyMemo_var1.wav"]
    dataset = calibrate_dataset(wav_files, len(wav_files))
    input_tensor = next(dataset).to(device)
    model_inference(model, q_layers, input_tensor)


if __name__ == '__main__':
    # 设备
    device = torch.device("cpu")
    # 打印选项设置为显示全部数据
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)

    # 输出日志
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    log_file = f'{out_dir}/log.txt'
    # 设置日志级别、输出格式和输出内容等
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format="%(message)s",
                        datefmt="%S")

    # 参数设置 #
    use_beco = True  # 是否使用BECO算子
    kws_class_num = 11  # 唤醒词数 + 1
    sample_num = 10000  # 模型量化标定样本数

    model_type = "kws"
    # 输出模型.h文件
    model_h_file = f'{out_dir}/nn_{model_type}_model.hpp'

    model_dir = "./models/TCN2/final_model_2"
    wavs_dir = "./datas/edgetts_generated/AUS_Sydney_Female_25_Fast"

    main(model_dir, wavs_dir, model_type, kws_class_num, sample_num, model_h_file, use_beco, device='cpu')
