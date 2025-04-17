import os
import shutil
import soundfile as sf
from tqdm import tqdm


def delete_unknown_files_and_folders(root_folder):
    """
    删除所有带有 'unknown' 的文件夹与文件

    参数:
    root_folder (str): 根文件夹路径
    """
    for root, dirs, files in os.walk(root_folder, topdown=False):
        # 删除文件
        for file in tqdm(files, desc=f"检查文件夹: {root}"):
            if 'unknown' in file.lower():
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件时出错: {file_path}, 错误: {e}")

        # 删除文件夹
        for dir in dirs:
            if 'unknown' in dir.lower():
                dir_path = os.path.join(root, dir)
                try:
                    shutil.rmtree(dir_path)
                    print(f"已删除文件夹: {dir_path}")
                except Exception as e:
                    print(f"删除文件夹时出错: {dir_path}, 错误: {e}")


def check_wav_file(wav_file):
    """
    检查WAV文件是否可以正常读取
    """
    try:
        sf.read(wav_file)
        return True
    except Exception:
        return False


def delete_invalid_wav_files(root_folder):
    """
    遍历文件夹及其子文件夹，删除不能被正常读取的WAV文件

    参数:
    root_folder (str): 根文件夹路径
    """
    for root, _, files in os.walk(root_folder):
        for file in tqdm(files, desc=f"检查文件夹: {root}"):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                if not check_wav_file(file_path):
                    try:
                        os.remove(file_path)
                        print(f"已删除无效文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件时出错: {file_path}, 错误: {e}")


# 使用示例
if __name__ == "__main__":
    root_folder = "./datas/unknown_data"  # 修改为实际的根文件夹路径
    # delete_unknown_files_and_folders(root_folder)
    delete_invalid_wav_files(root_folder)
