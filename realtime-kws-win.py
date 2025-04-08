#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时关键词检测系统 (Mac M4优化版)
================================

该程序使用麦克风实时检测关键词，支持两种检测模式:
1. 单帧检测模式：单次检测分数超过阈值即触发
2. 连续帧检测模式：连续多帧检测到同一关键词且分数均超过阈值时触发

适用于Mac M4处理器，可在IDE中直接运行
"""

import os
import logging
import torch
import numpy as np
import sounddevice as sd
import queue
import time
from datetime import datetime
from src.model.tcn import TCN2
from src.feature.featurizer import extract_feature


# ====================== 配置部分 ======================

class KwsConfig:
    """关键词检测系统配置类"""

    def __init__(self):
        # 关键词映射表 (索引:关键词)
        self.keywords = {
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
        }

        # 模型参数
        self.model_path = "./models/TCN2/last_model/model.pth"  # 模型文件路径
        self.num_class = 10  # 分类数量 (关键词数量+1个未知类)
        self.feature_dim = 40  # 特征维度

        # 检测参数
        self.detection_mode = 1  # 1=单帧检测, 2=连续帧检测
        self.threshold = 0.8  # 检测阈值 (0.0-1.0)
        self.consecutive_frames = 3  # 连续帧数量 (仅用于模式2)

        # 音频参数
        self.sample_rate = 16000  # 采样率
        self.window_size_ms = 2012  # 窗口大小(毫秒)
        self.padding_ms = 1000  # 填充大小(毫秒)
        self.frame_shift_ms = 20  # 帧移(毫秒)
        self.audio_device = None  # 音频设备索引(None=默认)
        self.block_size = 1600  # 音频处理块大小(设为采样率的十分之一)

        # 系统参数
        self.device = "auto"  # 计算设备: 'auto', 'cpu', 'mps' (Mac GPU), 'cuda'
        self.debug = True  # 是否启用调试日志
        self.log_file = ""  # 日志文件路径 (空字符串=不保存到文件)

        # 计算派生参数
        self._update_derived_params()

    def _update_derived_params(self):
        """更新派生参数"""
        # 计算音频相关的样本数量
        self.window_size = int(self.window_size_ms * self.sample_rate / 1000)
        self.padding_size = int(self.padding_ms * self.sample_rate / 1000)
        self.frame_shift = int(self.frame_shift_ms * self.sample_rate / 1000)
        self.total_buffer_size = self.window_size + 2 * self.padding_size

        # 提取特征参数
        self.fbank_args = {
            'sample_frequency': self.sample_rate,
            'num_mel_bins': self.feature_dim,
            'frame_length': 32.0,
            'frame_shift': self.frame_shift_ms,
            'high_freq': 0.0,
            'low_freq': 20.0,
            'preemphasis_coefficient': 0.0,
            'raw_energy': False,
            'remove_dc_offset': False,
            'use_power': True,
            'window_type': 'hanning'
        }


# ====================== 日志设置函数 ======================

def setup_logging(debug=False, log_file=None):
    """设置日志系统，可选文件输出和调试级别"""
    level = logging.DEBUG if debug else logging.INFO

    # 创建带时间戳的格式器
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 如果指定了日志文件，则创建文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"日志将同时保存到: {log_file}")


# ====================== 模型加载函数 ======================

def load_model(config):
    """加载关键词检测模型"""
    model_path = config.model_path
    model_dir = os.path.dirname(model_path)

    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        raise ValueError(f"模型文件 '{model_path}' 不存在")

    # 确定设备类型（CPU、Mac GPU 或 CUDA）
    if config.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("使用 Apple MPS (Metal Performance Shaders) 加速")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("使用 CUDA 加速")
        else:
            device = torch.device("cpu")
            logging.info("使用 CPU 进行推理")
    else:
        device = torch.device(config.device)

    # 创建模型实例
    model = TCN2(in_channels=config.feature_dim, num_class=config.num_class)

    # 加载模型参数
    logging.info(f"从 {model_path} 加载模型")
    try:
        model_state_dict = torch.load(model_path, map_location=device)
    # 处理状态字典中的键名，移除'_orig_mod.'前缀
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]  # 移除前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        model.eval()  # 设置为评估模式
        logging.info(f"模型加载成功，使用设备: {device}")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise

    return model, device


# ====================== 音频设备函数 ======================

def list_audio_devices():
    """列出所有可用的音频输入设备"""
    try:
        devices = sd.query_devices()
        logging.info("\n可用的音频设备:")
        logging.info("-" * 60)
        logging.info(f"{'索引':<6}{'输入通道':<10}{'采样率':<12}设备名称")
        logging.info("-" * 60)

        for i, device in enumerate(devices):
            input_channels = device['max_input_channels']
            if input_channels > 0:  # 只显示具有输入能力的设备
                samplerate = int(device['default_samplerate'])
                logging.info(f"[{i}]   {input_channels:<10}{samplerate:<12}{device['name']}")

        logging.info("-" * 60)
        return devices

    except Exception as e:
        logging.error(f"列出音频设备时出错: {str(e)}")
        return []


# 音频回调函数
def audio_callback(indata, frames, time_info, status):
    """音频回调函数，将捕获的音频数据放入队列"""
    if status:
        logging.warning(f"音频回调状态: {status}")
    try:
        audio_queue.put(indata.copy())
    except Exception as e:
        logging.error(f"音频回调错误: {str(e)}")


# ====================== 实时处理函数 ======================

def process_audio_stream(model, device, config):
    """实时处理音频流进行关键词检测"""
    # 常量定义
    UNKNOWN_INDEX = 0  # 未知词的索引

    # 显示音频设备信息
    current_device = sd.query_devices(config.audio_device)
    logging.info(f"使用音频设备: {current_device['name']}")

    # 音频缓冲区
    audio_buffer = np.zeros(config.total_buffer_size, dtype=np.float32)

    # 用于模式2（连续帧检测）
    detection_history = []

    # 性能跟踪
    inference_times = []
    last_report_time = time.time()

    logging.info("-" * 60)
    logging.info("开始实时关键词检测...")
    logging.info(f"检测模式: {config.detection_mode} ({'单帧检测' if config.detection_mode == 1 else '连续帧检测'})")
    logging.info(f"检测阈值: {config.threshold}")
    if config.detection_mode == 2:
        logging.info(f"连续帧数: {config.consecutive_frames}")
    logging.info(f"按 Ctrl+C 停止")
    logging.info("-" * 60)

    # 启动音频输入流
    try:
        stream = sd.InputStream(
            device=config.audio_device,
            channels=1,
            samplerate=config.sample_rate,
            blocksize=config.block_size,
            callback=audio_callback
        )

        stream.start()
        logging.info("音频流启动成功")

    except Exception as e:
        logging.error(f"启动音频流失败: {str(e)}")
        return

    try:
        while True:
            # 从队列获取数据
            try:
                new_data = audio_queue.get(timeout=0.1).flatten()

                # 移动缓冲区并添加新数据
                audio_buffer = np.roll(audio_buffer, -len(new_data))
                audio_buffer[-len(new_data):] = new_data

                # 准备输入张量
                input_tensor = torch.tensor(audio_buffer, dtype=torch.float32).unsqueeze(0).to(device)

                # 测量推理时间
                start_time = time.time()

                # 提取特征
                feature = extract_feature(input_tensor, config.fbank_args, device)

                # 前向传播
                with torch.no_grad():
                    output = model(feature)
                    output = torch.nn.functional.softmax(output, dim=-1)

                # 计算推理时间
                inference_time = (time.time() - start_time) * 1000  # 毫秒
                inference_times.append(inference_time)

                # 定期报告平均推理时间
                current_time = time.time()
                if current_time - last_report_time > 10.0:  # 每10秒报告一次
                    if inference_times:
                        avg_time = sum(inference_times) / len(inference_times)
                        logging.info(f"性能: {avg_time:.2f} 毫秒/推理, {len(inference_times) / 10:.1f} 推理/秒")
                        inference_times = []
                        last_report_time = current_time

                # 获取预测结果
                output_numpy = output.cpu().numpy()[0]
                index = np.argmax(output_numpy)
                score = output_numpy[index]

                # 获取关键词名称
                keyword = config.keywords.get(index, f"类别_{index}")

                # 调试日志
                if config.debug:
                    # 记录前3个预测结果及其分数
                    top_indices = np.argsort(output_numpy)[-3:][::-1]
                    prediction_str = ", ".join([f"{config.keywords.get(i, f'类别_{i}')}: {output_numpy[i]:.3f}"
                                                for i in top_indices])
                    logging.debug(f"前3预测: {prediction_str}")

                # 模式1：单帧检测
                if config.detection_mode == 1:
                    if index != UNKNOWN_INDEX and score > config.threshold:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        logging.info(f"[{timestamp}] 检测到关键词: {keyword} (索引={index}), 分数={score:.3f}")

                        # 重置缓冲区 - 这是实时处理的关键部分
                        audio_buffer = np.zeros_like(audio_buffer)

                # 模式2：连续帧检测
                else:
                    # 记录本次检测
                    detection_history.append((index, score))

                    # 只保留最后N帧用于分析
                    if len(detection_history) > config.consecutive_frames:
                        detection_history.pop(0)

                    # 检查是否有足够的历史记录
                    if len(detection_history) == config.consecutive_frames:
                        # 检查所有帧是否检测到相同的关键词且超过阈值
                        keyword_indices = [det[0] for det in detection_history]
                        scores = [det[1] for det in detection_history]

                        # 所有索引相同且不是未知词，所有分数都超过阈值
                        if (all(idx == keyword_indices[0] for idx in keyword_indices) and
                                keyword_indices[0] != UNKNOWN_INDEX and
                                all(s > config.threshold for s in scores)):
                            avg_score = sum(scores) / len(scores)
                            detected_keyword = config.keywords.get(keyword_indices[0], f"类别_{keyword_indices[0]}")

                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            logging.info(
                                f"[{timestamp}] 检测到关键词: {detected_keyword} (索引={keyword_indices[0]}), 平均分数={avg_score:.3f}")

                            # 重置缓冲区和历史记录 - 用于实时处理
                            audio_buffer = np.zeros_like(audio_buffer)
                            detection_history = []

            except queue.Empty:
                continue

            except Exception as e:
                logging.error(f"处理音频时出错: {str(e)}", exc_info=config.debug)
                continue

    except KeyboardInterrupt:
        logging.info("停止关键词检测...")
    finally:
        # 停止并关闭音频流
        try:
            stream.stop()
            stream.close()
            logging.info("音频流已关闭")
        except Exception as e:
            logging.error(f"关闭音频流时出错: {str(e)}")


# ====================== 主函数 ======================
def main():
    """主函数"""
    # 创建配置
    config = KwsConfig()

    # 设置日志
    log_file = ""
    if config.debug:
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"kws_realtime_{timestamp}.log"

    setup_logging(debug=config.debug, log_file=log_file)

    # 打印系统信息
    logging.info(f"实时关键词检测系统 - Windows 优化版")
    logging.info(f"操作系统: {os.name.upper()}")
    logging.info(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")  # 移除MPS相关显示

    # 列出音频设备
    devices = list_audio_devices()

    # 默认选择第一个输入设备（如果未指定）
    if config.audio_device is None and devices:
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                config.audio_device = i
                logging.info(f"自动选择设备索引: {i}")
                break

    try:
        # 加载模型
        model, device = load_model(config)

        # 创建音频数据队列
        global audio_queue
        audio_queue = queue.Queue()

        # 开始实时处理
        process_audio_stream(model, device, config)

    except Exception as e:
        logging.error(f"运行时错误: {str(e)}", exc_info=True)


# ====================== 调用入口点 ======================

if __name__ == "__main__":
    main()
