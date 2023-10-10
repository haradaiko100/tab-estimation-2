from datetime import datetime
import pandas as pd
from move import get_same_note_nodes
import numpy as np
import librosa
import librosa.display
from matplotlib import lines as mlines, pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from predict import TDR, calculate_metrics, tab2pitch
import seaborn as sns
import librosa
import torch
import yaml
import os
import argparse
import glob
from multiprocessing import Pool
from itertools import repeat
import tqdm

import networkx as nx


def save_npz_notes(npz_file_path, graph_tab_data, note_F0_from_tab_graph_pred):
    # 既存のnpzファイルを読み込む
    existing_data = np.load(npz_file_path)

    # 既存のデータを新しい辞書にコピー
    new_data = {key: existing_data[key] for key in existing_data.files}

    # 新しいデータを追加する
    new_data["note_tab_graph_pred"] = graph_tab_data
    new_data["note_F0_from_tab_graph_pred"] = note_F0_from_tab_graph_pred

    # 新しい辞書を含む新しいnpzファイルを保存する
    np.savez_compressed(npz_file_path, **new_data)

    return


def get_fingers_distance_dict(note: np.ndarray):
    # 各弦の押している位置を取得
    finger_positions = np.argmax(note, axis=1)

    # finger_positionsから開放弦(インデックス：0)と鳴らしてない弦(インデックス：20)の情報を削除
    new_finger_positions = [
        elem for elem in finger_positions if elem != 0 and elem != 20
    ]

    # 要素ない場合は0を返す
    if len(new_finger_positions) == 0:
        return {
            "max": 0,
            "min": 0,
        }

    max_index = max(new_finger_positions)
    min_index = min(new_finger_positions)

    return {
        "max": max_index,
        "min": min_index,
    }


# カウントの情報とdataのnumpy配列を返す
def get_same_time_nodes_from_graph(graph, target_time):
    # same_time_nodes = np.empty([0,0,0])
    same_time_nodes_info = []
    for node in graph.nodes():
        if "time" in graph.nodes[node] and graph.nodes[node]["time"] == target_time:
            node_chord_info = graph.nodes[node]["data"]

            same_time_nodes_info.append(
                {
                    "data": node_chord_info,
                    "count": node,
                }
            )

    return same_time_nodes_info


def calc_weight_between_notes(prev_note: np.ndarray, current_note: np.ndarray):
    current_fingers_dict = get_fingers_distance_dict(current_note)
    prev_fingers_dict = get_fingers_distance_dict(prev_note)
    weight = 0

    current_fingers_distance = current_fingers_dict["max"] - current_fingers_dict["min"]

    current_mid_fret = sum(current_fingers_dict.values()) / len(current_fingers_dict)

    prev_fret = sum(prev_fingers_dict.values()) / len(prev_fingers_dict)

    # 開放弦のみ or 弦引いてないときは重みを0にする
    if current_fingers_dict["max"] == 0 and current_fingers_dict["min"] == 0:
        return weight

    else:
        # 前の音が開放弦 or ミュートの時
        if prev_fret == 0:
            # ここのweightが一つの弦だけを押してるときに、
            # current_fingers_distanceだけだと遠い弦と近い弦の差が出なくなっちゃう
            weight = current_fingers_distance + current_fingers_dict["max"]

        else:
            weight = abs(prev_fret - current_mid_fret) + (current_fingers_distance)
        if current_fingers_dict["max"] > 10:
            weight += 1

    return weight


def estimate_tab_from_pred(tab: np.ndarray):
    DG = nx.DiGraph()
    current_time = 0
    prev_time = 0
    node_count = 1
    dest_node_count = 1

    current_stopping_point_index = 1
    next_stopping_point_index = 1

    stopping_point_node_list = []
    shortest_path_list = []

    for note_index, note in enumerate(tab):
        if current_time == 0:
            DG.add_node(node_count, data=note, time=current_time, count=node_count)
            node_count += 1

        else:
            # get_same_note_nodes内でありえないノードは含まれないようにしたい
            same_note_nodes = get_same_note_nodes(note)  # numpyの配列
            prev_time_nodes = get_same_time_nodes_from_graph(DG, prev_time)

            # 小節の区切りに当たるCNNからのノードを記録
            if (note_index + 1) % 16 == 0:
                stopping_point_node_list.append(node_count)

            # 最短経路の目的地のノードを設定する
            if note_index == len(tab) - 1:
                dest_node_count = node_count

            # グラフにノード追加
            for i in range(len(same_note_nodes)):
                DG.add_node(
                    node_count,
                    data=same_note_nodes[i],
                    time=current_time,
                    count=node_count,
                )

                # エッジ追加
                for current_node in same_note_nodes:
                    for prev_node in prev_time_nodes:
                        weight = calc_weight_between_notes(
                            current_note=current_node, prev_note=prev_node["data"]
                        )
                        DG.add_edge(prev_node["count"], node_count, weight=weight)

                # エッジ追加後にインクリメント
                node_count += 1

        # 時間をインクリメント
        prev_time = current_time
        current_time += 1

    # 小節ごとに最短経路を求める
    for i in range(len(stopping_point_node_list)):
        each_shortest_path = nx.dijkstra_path(
            G=DG,
            source=current_stopping_point_index,
            target=stopping_point_node_list[i],
            weight="weight",
        )
        # shortest_path_list.append(each_shortest_path)
        shortest_path_list += each_shortest_path

        current_stopping_point_index = stopping_point_node_list[i] + 1

    # 重複あったら削除する
    shortest_path_list = list(set(shortest_path_list))

    # shortest_path = nx.dijkstra_path(
    #     G=DG, source=1, target=dest_node_count, weight="weight"
    # )

    # shortest_pathの実際のデータを取得する
    estimated_tab = [DG.nodes[node]["data"] for node in shortest_path_list]

    # ndarrayに変換
    npz_estimated_tab = np.array(estimated_tab)

    return npz_estimated_tab


def estimate_and_save_tab_in_npz(npz_filename_list, test_num):
    npz_filename_list = npz_filename_list.split("\n")

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

    for npz_file in npz_filename_list:
        # ex) npz_file: result/tab/202304241804_epoch192/npz/test_02/02_Funk1-97-C_comp_01.npz
        npz_data = np.load(npz_file)
        note_pred = npz_data["note_tab_pred"]
        note_gt = npz_data["note_tab_gt"]

        frame_gt = npz_data["frame_tab_gt"]
        frame_pred = npz_data["frame_tab_pred"]

        note_F0_gt = npz_data["note_F0_gt"]
        frame_F0_gt = npz_data["frame_F0_gt"]

        estimated_tab = estimate_tab_from_pred(note_pred)
        note_F0_from_tab_graph_pred = tab2pitch(estimated_tab)

        frame_tdr = TDR(frame_pred, frame_gt, frame_F0_gt)
        note_tdr = TDR(note_pred, note_gt, note_F0_gt)

        frame_note_tdr = TDR(frame_pred, frame_gt, frame_F0_gt)
        note_graph_tdr = TDR(estimated_tab, note_gt, note_F0_gt)

        frame_sum_tdr += frame_tdr
        note_sum_tdr += note_tdr

        frame_graph_sum_tdr += frame_note_tdr
        note_graph_sum_tdr += note_graph_tdr

        save_npz_notes(
            npz_file_path=npz_file,
            graph_tab_data=estimated_tab,
            note_F0_from_tab_graph_pred=note_F0_from_tab_graph_pred,
        )

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
    note_graph_avg_tdr = note_graph_sum_tdr / len(npz_filename_list)

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

    mode = "tab"
    n_cores = 12

    if mode == "F0":
        npz_dir = os.path.join(
            "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )
    elif mode == "tab":
        npz_dir = os.path.join(
            "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )

    now = datetime.now()
    now_formated = now.strftime("%Y%m%d_%H%M%S")  # "%d/%m/%Y %H:%M:%S"
    # print("Today's date: ", today_formated)

    metrics_data = pd.DataFrame()
    result_path = os.path.join("result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    csv_path = os.path.join(
        result_path,
        f"{mode}_graph",
        trained_model + f"_epoch{use_model_epoch}",
        now_formated,
        "metrics.csv",
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
                "tab_graph",
                f"{trained_model}_epoch{use_model_epoch}",
                now_formated,
                "visualize",
                f"test_0{test_num}",
            )

        npz_filename_list = glob.glob(os.path.join(npz_dir, f"test_0{test_num}", "*"))
        if isinstance(npz_filename_list, list):
            npz_filename_list = "\n".join(npz_filename_list)
        # kwargs["visualize_dir"] = visualize_dir
        if not (os.path.exists(visualize_dir)):
            os.makedirs(visualize_dir)

        metrics_result = estimate_and_save_tab_in_npz(
            npz_filename_list=npz_filename_list, test_num=test_num
        )

        metrics_data = pd.concat([metrics_data, metrics_result], axis=0)

    metrics_data = metrics_data.append(metrics_data.describe()[1:3])
    metrics_data.to_csv(csv_path, float_format="%.3f")


if __name__ == "__main__":
    main()
