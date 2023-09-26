import os

def count_files_in_directory(directory_path):
    try:
        file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        print(f"Number of files in directory '{directory_path}': {file_count}")
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# ディレクトリのパスを指定してファイル数を出力
directory_path = "/home/yuki/university/research/Tab-estimator2/GuitarSet/audio_mono-mic"
# /home/yuki/university/research/Tab-estimator2/GuitarSet/audio_mono-mic
count_files_in_directory(directory_path)