# 语音文件命名格式说明

## 文件命名规范

语音数据集中的文件采用以下命名规范：

```
SPK001_AUS_Sydney_male_40_HeyMemo_-15.4dB_1.7wps.wav
```

## 目录结构

1. 文件在以SPKID命名的子文件夹中：

```
根目录/train_data_resampled/
  ├── SPK001_resampled/
  │     ├── SPK001_AUS_Sydney_male_40_HeyMemo_-15.4dB_1.7wps.wav
  │     ├── SPK001_AUS_Sydney_male_40_HeyMemo_-10.2dB_1.5wps.wav
  │     └── ...
  ├── SPK002_resampled/
  │     ├── SPK002_CAN_LONDON_MALE_29_HeyMemo_-20.9dB_1.8wps.wav
  │     └── ...
  └── ...
```

2. edgetts文件:

```
/edgetts_generated/
  ├── AUS_Sydney_Female_25_Fast/
  │     ├── AUS_Sydney_Female_25_HeyMemo_var1.wav
  │     ├── AUS_Sydney_Female_25_HeyMemo_var2.wav
  │     └── ...
  ├── AUS_Sydney_Female_25_Normal/
  │     ├── AUS_Sydney_Female_25_HeyMemo_var1.wav
  │     └── ...
  └── ...
```

## 命名格式解析

每个文件名由多个字段组成，以下划线(`_`)分隔：

| 字段位置 | 字段名称 | 示例值 | 描述 |
|---------|---------|-------|------|
| 1 | SPKID | SPK001 | 格式为"SPK"后跟三位数字 |
| 2 | 国家/地区 | AUS | 国家或地区 |
| 3 | 城市 | Sydney | 所在城市 |
| 4 | 性别 | male | 性别（male/female） |
| 5 | 年龄 | 40 | 年龄                                     |
| 6 | 唤醒词/命令词 | HeyMemo | 关键词 |
| 7 | 信噪比 | -15.4dB | 录音的音量，以分贝(dB)为单位 |
| 8 | 语速 | 1.7wps | 语速，以每秒词数(words per second)为单位 |

## 文件格式

- **文件类型**：.wav
- **采样率**：16kHz
- **比特深度**：16位
- **声道**：单声道
- **唤醒词/命令词**："HeyMemo", "TakeAPicture", "TakeAVideo", "StopRecording", "Pause", "Next", "Play", "VolumeUp", "VolumeDown", "Unknown"

