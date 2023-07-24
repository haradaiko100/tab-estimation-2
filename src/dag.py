from move import get_same_note_nodes
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
import tqdm

import networkx as nx


def save_npz_notes(test_num, graph_tab_data):
    # results/tabにあるデータを読み込んでそこに新しいデータを追加するようにしたい

    test_data_path = os.path.join(
        "data", "npz", f"original", "split", f"0{test_num}_*.npz"
    )
    test_data_list = np.array(glob.glob(test_data_path, recursive=True))
    for npz_filename in tqdm.tqdm(test_data_list):
        npz_save_dir = os.path.join(
            "result",
            "tab",
            f"{trained_model}_epoch{use_model_epoch}",
            "npz",
            f"test_0{test_num}",
        )
        npz_save_filename = os.path.join(npz_save_dir, os.path.split(npz_filename)[1])
        if not (os.path.exists(npz_save_dir)):
            os.makedirs(npz_save_dir)

        # 既存のnpzファイルを読み込む
        existing_data = np.load(npz_save_filename)

        # 新しいデータを追加する
        existing_data["note_tab_graph_pred"] = graph_tab_data

        # 追加したデータを含む新しいnpzファイルを保存する
        np.savez_compressed(npz_save_filename, **existing_data)

    return


def get_fingers_distance_dict(note: np.ndarray):
    # 各弦の押している位置を取得
    finger_positions = np.argmax(note, axis=1)

    # finger_positionsから開放弦(0)と鳴らしてない弦(21)の情報を削除
    new_finger_positions = [
        elem for elem in finger_positions if elem != 0 and elem != 21
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

    prev_fret = sum(prev_fingers_dict.values()) / len(prev_fingers_dict)

    # 開放弦のみ or 弦引いてないときは重みを0にする
    if current_fingers_dict["max"] == 0 and current_fingers_dict["min"] == 0:
        weight = 0

    else:
        weight = abs(prev_fret - (current_fingers_distance / 2)) + (
            current_fingers_distance
        )
        if current_fingers_dict["max"] > 7:
            weight += 1

    return weight


def estimate_tab_from_pred(tab: np.ndarray):
    DG = nx.DiGraph()
    current_time = 0
    prev_time = 0
    node_count = 1

    for note in tab:
        if current_time == 0:
            DG.add_node(node_count, data=note, time=current_time, count=node_count)
            node_count += 1

        else:
            # get_same_note_nodes内でありえないノードは含まれないようにしたい
            same_note_nodes = get_same_note_nodes(note)  # numpyの配列
            prev_time_nodes = get_same_time_nodes_from_graph(DG, prev_time)

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

    shortest_path = nx.dijkstra_path(
        G=DG, source=1, target=node_count - 1, weight="weight"
    )

    # shortest_pathの実際のデータを取得する
    estimated_tab = [DG.nodes[node]["data"] for node in shortest_path]

    # ndarrayに変換
    npz_estimated_tab = np.array(estimated_tab)

    save_npz_notes(npz_estimated_tab)
    return 


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
                "tab_graph",
                f"{trained_model}_epoch{use_model_epoch}",
                "visualize",
                f"test_0{test_num}",
            )

        npz_filename_list = glob.glob(os.path.join(npz_dir, f"test_0{test_num}", "*"))
        kwargs["visualize_dir"] = visualize_dir
        if not (os.path.exists(visualize_dir)):
            os.makedirs(visualize_dir)

        # paralell process
        p = Pool(n_cores)

        # npz_filename_list：result/tab/npzのとこにあるnpzのファイル
        # npz_filename_listの全てのファイルに対してvisualizeを実行している
        p.starmap(estimate_tab_from_pred, zip(npz_filename_list, repeat(kwargs)))
        p.close()  # or p.terminate()
        p.join()


if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description="code for plotting results")
    # parser.add_argument(
    #     "model", type=str, help="name of trained model: ex) 202201010000"
    # )
    # parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     help="option for verbosity: -v to turn on verbosity",
    #     action="store_true",
    #     required=False,
    #     default=False,
    # )
    # args = parser.parse_args()

    # trained_model = args.model
    # use_model_epoch = args.epoch
    # npz_dir = os.path.join(
    #     "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
    # )
    # # npz_filename_list = glob.glob(
    # #         os.path.join(npz_dir, f"test_0{test_num}", "*"))
    # npz_filename_list = glob.glob(os.path.join(npz_dir, "test_00", "*"))

    # npz_data = np.load(npz_filename_list[3])
    # # print(npz_filename_list[3])
    # note_pred = npz_data["note_tab_pred"]

    # estimated_tab = estimate_tab_from_pred(note_pred)
    # print(estimated_tab)
