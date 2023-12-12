import glob
import numpy as np
import os
import torch
import pandas as pd
import tqdm
import yaml
from network import TabEstimator
from visualize import visualize
import tqdm
from matplotlib import lines as mlines, pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
import argparse
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(pred, gt):
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary"
    )

    return precision, recall, f1


def tab2pitch(tab):
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    argmax_index = np.argmax(tab, axis=2)
    pitch = np.zeros((len(tab), 44))
    for time in range(len(tab)):
        for string in range(6):
            if argmax_index[time, string] < 20:
                pitch[time, argmax_index[time, string] + rel_string_pitches[string]] = 1

    return pitch


def TDR(tab_pred, tab_gt, F0_gt):
    F0_from_tab_pred = tab2pitch(tab_pred)

    TP_tab = np.multiply(tab_gt[:, :, :-1], tab_pred[:, :, :-1]).sum()
    TP_F0 = np.multiply(F0_gt, F0_from_tab_pred).sum()

    if TP_F0 == 0:
        raise ValueError("TP_F0 is 0, cannot calculate TDR")
    tdr = TP_tab / TP_F0
    return tdr


def calc_score(
    test_num,
    trained_model,
    use_model_epoch,
    date,
):
    (
        frame_sum_F0_from_tab_precision,
        frame_sum_F0_from_tab_recall,
        frame_sum_F0_from_tab_f1,
    ) = (0, 0, 0)

    (
        note_sum_F0_from_tab_precision,
        note_sum_F0_from_tab_recall,
        note_sum_F0_from_tab_f1,
    ) = (0, 0, 0)

    (
        note_sum_F0_from_tab_graph_precision,
        note_sum_F0_from_tab_graph_recall,
        note_sum_F0_from_tab_graph_f1,
    ) = (0, 0, 0)
    frame_sum_tdr, note_sum_tdr = 0, 0
    frame_graph_sum_tdr, note_graph_sum_tdr = 0, 0

    frame_sum_precision, frame_sum_recall, frame_sum_f1 = 0, 0, 0
    note_sum_precision, note_sum_recall, note_sum_f1 = 0, 0, 0

    frame_concat_pred = np.array([])
    frame_concat_gt = np.array([])

    npz_filename_path = os.path.join(
        "result",
        "same_sound_issue_data",
        f"{trained_model}_epoch{use_model_epoch}",
        date,
        "npz",
        f"test_0{test_num}",
        f"0{test_num}_*.npz",
    )
    npz_filename_list = np.array(glob.glob(npz_filename_path, recursive=True))
    # every test file for loop start
    frame_concat_pred = np.array([])
    frame_concat_gt = np.array([])

    file_num = len(npz_filename_list)

    for npz_file in npz_filename_list:
        npz_data = np.load(npz_file)
        note_pred = npz_data["note_tab_pred"]
        note_gt = npz_data["note_tab_gt"]

        frame_gt = npz_data["frame_tab_gt"]
        frame_pred = npz_data["frame_tab_pred"]

        note_F0_gt = npz_data["note_F0_gt"]
        frame_F0_gt = npz_data["frame_F0_gt"]

        # 異弦同音関連のデータ
        same_sound_issue_tab = npz_data["same_sound_issue_tab"]
        same_sound_issue_pred_tab = npz_data["same_sound_issue_pred_tab"]
        same_sound_issue_graph_tab = npz_data["same_sound_issue_graph_tab"]

        # 異弦同音だけのデータであれば、CNNの方のTDRは必ず0になるから
        # グラフの方のTDRを計算するだけでいい
        note_F0_from_tab_graph_pred = tab2pitch(same_sound_issue_graph_tab)

        frame_tdr = TDR(frame_pred, frame_gt, frame_F0_gt)
        note_tdr = TDR(note_pred, note_gt, note_F0_gt)

        try:
            note_F0_gt_for_same_sound_issue = tab2pitch(same_sound_issue_tab)
            note_graph_tdr = TDR(
                same_sound_issue_graph_tab,
                same_sound_issue_tab,
                note_F0_gt_for_same_sound_issue,
            )

            note_graph_sum_tdr += note_graph_tdr
        except ValueError:
            file_num -= 1
        frame_note_tdr = TDR(frame_pred, frame_gt, frame_F0_gt)

        frame_sum_tdr += frame_tdr
        note_sum_tdr += note_tdr

        frame_graph_sum_tdr += frame_note_tdr

        frame_pred = frame_pred[:, :, :-1].flatten()
        frame_gt = frame_gt[:, :, :-1].flatten()
        note_pred = note_pred[:, :, :-1].flatten()
        note_gt = note_gt[:, :, :-1].flatten()

        note_F0_from_tab_graph_pred = note_F0_from_tab_graph_pred.flatten()

        note_F0_gt = note_F0_gt.flatten()

        frame_concat_pred = np.concatenate((frame_concat_pred, frame_pred), axis=None)
        frame_concat_gt = np.concatenate((frame_concat_gt, frame_gt), axis=None)

        frame_precision, frame_recall, frame_f1 = calculate_metrics(
            frame_pred, frame_gt
        )
        note_precision, note_recall, note_f1 = calculate_metrics(note_pred, note_gt)

        (
            note_F0_from_tab_graph_precision,
            note_F0_from_tab_graph_recall,
            note_F0_from_tab_graph_f1,
        ) = calculate_metrics(note_F0_from_tab_graph_pred, note_F0_gt)

        frame_sum_precision += frame_precision
        frame_sum_recall += frame_recall
        frame_sum_f1 += frame_f1

        note_sum_precision += note_precision
        note_sum_recall += note_recall
        note_sum_f1 += note_f1

        note_sum_F0_from_tab_graph_precision += note_F0_from_tab_graph_precision
        note_sum_F0_from_tab_graph_recall += note_F0_from_tab_graph_recall
        note_sum_F0_from_tab_graph_f1 += note_F0_from_tab_graph_f1

        print(f"finished {os.path.split(npz_file)[1][:-4]}")

    frame_avg_precision = frame_sum_precision / len(npz_filename_list)
    frame_avg_recall = frame_sum_recall / len(npz_filename_list)
    frame_avg_f1 = frame_sum_f1 / len(npz_filename_list)

    note_avg_precision = note_sum_precision / len(npz_filename_list)
    note_avg_recall = note_sum_recall / len(npz_filename_list)
    note_avg_f1 = note_sum_f1 / len(npz_filename_list)

    frame_avg_F0_from_tab_precision = frame_sum_F0_from_tab_precision / len(
        npz_filename_list
    )
    frame_avg_F0_from_tab_recall = frame_sum_F0_from_tab_recall / len(npz_filename_list)
    frame_avg_F0_from_tab_f1 = frame_sum_F0_from_tab_f1 / len(npz_filename_list)

    note_avg_F0_from_tab_graph_precision = note_sum_F0_from_tab_graph_precision / len(
        npz_filename_list
    )
    note_avg_F0_from_tab_graph_recall = note_sum_F0_from_tab_graph_recall / len(
        npz_filename_list
    )
    note_avg_F0_from_tab_graph_f1 = note_sum_F0_from_tab_graph_f1 / len(
        npz_filename_list
    )

    frame_avg_tdr = frame_sum_tdr / len(npz_filename_list)
    note_avg_tdr = note_sum_tdr / len(npz_filename_list)

    frame_graph_avg_tdr = frame_graph_sum_tdr / len(npz_filename_list)
    note_graph_avg_tdr = note_graph_sum_tdr / file_num

    frame_concat_precision, frame_concat_recall, frame_concat_f1 = calculate_metrics(
        frame_concat_pred, frame_concat_gt
    )

    result = pd.DataFrame(
        [
            [
                frame_avg_precision,
                frame_avg_recall,
                frame_avg_f1,
                frame_concat_precision,
                frame_concat_recall,
                frame_concat_f1,
                note_avg_precision,
                note_avg_recall,
                note_avg_f1,
                frame_avg_F0_from_tab_precision,
                frame_avg_F0_from_tab_recall,
                frame_avg_F0_from_tab_f1,
                note_avg_F0_from_tab_graph_precision,
                note_avg_F0_from_tab_graph_recall,
                note_avg_F0_from_tab_graph_f1,
                frame_avg_tdr,
                note_avg_tdr,
                frame_graph_avg_tdr,
                note_graph_avg_tdr,
            ]
        ],
        columns=[
            "frame_segment_avg_tab_p",
            "frame_segment_avg_tab_r",
            "frame_segment_avg_tab_f",
            "frame_frame_avg_tab_p",
            "frame_frame_avg_tab_r",
            "frame_frame_avg_tab_f",
            "note_avg_tab_p",
            "note_avg_tab_r",
            "note_avg_tab_f",
            "frame_avg_F0_from_tab_p",
            "frame_avg_F0_from_tab_r",
            "frame_avg_F0_from_tab_f",
            "note_avg_F0_from_tab_graph_p",
            "note_avg_F0_from_tab_graph_r",
            "note_avg_F0_from_tab_graph_f",
            "frame_avg_tdr",
            "note_avg_tdr",
            "frame_graph_avg_tdr",
            "note_graph_avg_tdr",
        ],
        index=[f"No0{test_num}"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="code for predicting and saving results"
    )
    parser.add_argument(
        "model", type=str, help="name of trained model: ex) 202201010000"
    )
    parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    parser.add_argument(
        "directory_date", type=str, help="date of dag.py was carried out"
    )  # dag.pyでできたディレクトリの日付を指定する
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
    date = args.directory_date
    verbose = args.verbose

    input_as_random_noize = False
    plot_results = False
    make_notelvl_from_framelvl = False

    result = pd.DataFrame()
    config_path = os.path.join("model", f"{trained_model}", "config.yaml")

    with open(config_path) as f:
        obj = yaml.safe_load(f)
        mode = obj["mode"]

    csv_path = os.path.join(
        "result",
        "same_sound_issue_data",
        f"{trained_model}_epoch{use_model_epoch}",
        date,
        "issue_metrics.csv",
    )

    for test_num in range(6):
        print(f"Player No. {test_num}")
        result = result.append(
            calc_score(
                test_num,
                trained_model,
                use_model_epoch,
                date=date,
            )
        )
    result = result.append(result.describe()[1:3])
    result.to_csv(csv_path, float_format="%.3f")
    return


if __name__ == "__main__":
    main()
