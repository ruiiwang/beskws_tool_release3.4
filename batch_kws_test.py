import os
import subprocess

def batch_run_kws(wav_dir, threshold, model_hpp, log_file="kws_test_history.log"):
    # 获取所有wav文件（递归查找）
    wav_files = []
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    # 创建日志文件头
    with open(log_file, 'a') as f:
        f.write(f"参数设置: 阈值={threshold}, 模型={model_hpp}\n")
    
    for wav_path in wav_files:
        cmd = f"CPP_simu/kws_test {wav_path} {threshold} {model_hpp}"
        
        # 执行命令并捕获输出
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"\n处理文件: {wav_path}\n")
            f.write(f"输出结果:\n{result.stdout}")
            if result.stderr:
                f.write(f"错误信息:\n{result.stderr}\n")

if __name__ == '__main__':
    # 参数设置
    wav_directory = "datas/unknown_data"  # 修改为双层目录
    threshold_value = "0.9"  # 唤醒阈值
    model_file = "output/nn_kws_model.hpp"  # 模型文件
    
    batch_run_kws(wav_directory, threshold_value, model_file)
    