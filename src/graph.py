import numpy as np
import librosa
import librosa.display
from matplotlib import lines as mlines, pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
import yaml
import os
import argparse
import glob
from multiprocessing import Pool
from itertools import repeat

import heapq
import sys


def main():
    parser = argparse.ArgumentParser(description="code for plotting results")
    parser.add_argument(
        "model", type=str, help="name of trained model: ex) 202201010000"
    )
    parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    parser.add_argument(
        "-v",
        "--verbose",
        help="option for verbosity: -v to turn on verbosity",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    verbose = args.verbose

    config_path = os.path.join("model", f"{trained_model}", "config.yaml")
    with open(config_path) as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]
        down_sampling_rate = obj["down_sampling_rate"]
        bins_per_octave = obj["bins_per_octave"]
        hop_length = obj["hop_length"]
        encoder_layers = obj["encoder_layers"]
        encoder_heads = obj["encoder_heads"]
        n_cores = obj["n_cores"]
        mode = obj["mode"]
        input_feature_type = obj["input_feature_type"]

    kwargs = {
        "note_resolution": note_resolution,
        "down_sampling_rate": down_sampling_rate,
        "bins_per_octave": bins_per_octave,
        "hop_length": hop_length,
        "encoder_layers": encoder_layers,
        "encoder_heads": encoder_heads,
        "mode": mode,
        "input_feature_type": input_feature_type,
    }

    if mode == "F0":
        npz_dir = os.path.join(
            "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )
    elif mode == "tab":
        npz_dir = os.path.join(
            "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )

    for test_num in range(6):
        if mode == "F0":
            visualize_dir = os.path.join(
                "result",
                "F0",
                f"{trained_model}_epoch{use_model_epoch}",
                "visualize",
                f"test_0{test_num}",
            )
        if mode == "tab":
            visualize_dir = os.path.join(
                "result",
                "tab",
                f"{trained_model}_epoch{use_model_epoch}",
                "visualize",
                f"test_0{test_num}",
            )

        npz_filename_list = glob.glob(os.path.join(npz_dir, f"test_0{test_num}", "*"))
        kwargs["visualize_dir"] = visualize_dir
        # if not(os.path.exists(visualize_dir)):
        #     os.makedirs(visualize_dir)

        # # paralell process
        # p = Pool(n_cores)
        # p.starmap(visualize, zip(npz_filename_list, repeat(kwargs)))
        # p.close()  # or p.terminate()
        # p.join()


# 1曲のタブ譜の各フレームにおいて、なっている音の高さ(pitch)を算出する
def tab2pitch(tab):
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    argmax_index = np.argmax(tab, axis=2)
    pitch = np.zeros((len(tab), 44))
    for time in range(len(tab)):
        for string in range(6):
            if argmax_index[time, string] < 20:
                pitch[time, argmax_index[time, string] + rel_string_pitches[string]] = 1

    return pitch


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description="code for plotting results")
    parser.add_argument(
        "model", type=str, help="name of trained model: ex) 202201010000"
    )
    parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    parser.add_argument(
        "-v",
        "--verbose",
        help="option for verbosity: -v to turn on verbosity",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    npz_dir = os.path.join(
        "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
    )
    # npz_filename_list = glob.glob(
    #         os.path.join(npz_dir, f"test_0{test_num}", "*"))
    npz_filename_list = glob.glob(os.path.join(npz_dir, "test_00", "*"))

    npz_data = np.load(npz_filename_list[2])
    print(npz_filename_list[2])
    note_pred = npz_data["note_tab_pred"]
    print(npz_data["note_tab_gt"].shape)
    # frame_pred = npz_data["frame_tab_pred"]
    # print(frame_pred.shape)
    print(note_pred.shape)
    # print(note_pred[20])
    # note_f0 = npz_data["note_F0_from_tab_pred"] # 2次元配列 64×44形式 44はギターで出せる音の高さの数
    # print(note_f0[5])
    # result = tab2pitch(note_pred)
    # print(result.shape)
    # print(result[20])
    # print(note_pred[3].shape)
    # print(note_pred.shape)
