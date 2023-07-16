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

class Node:
    def __init__(self, string, fret, cost):
        self.string = string  # 弦の番号
        self.fret = fret  # フレットの番号
        self.cost = cost  # スタートノードからのコスト

def check_right_column_zeros(array):
    right_column = array[:, -1]  # 右端の列を取得
    return np.all(right_column == 0)  # 右端の列がすべて0かどうかを判定


def get_start_node_index(tab):
    for i in range(len(tab)):
        if check_right_column_zeros(tab[i]) == False:
            return i

def get_fingers_distance(note):
    finger_positions = np.argmax(note,axis=1)

    

def dijkstra(graph,start_node_index):
    num_nodes = len(graph) # ここ後で変える
    visited = [False] * num_nodes
    distances = np.inf * np.ones(num_nodes)
    distances[0] = 0

    # 3次元の空配列を作成
    converted_tab = np.empty([0,0,0])

    queue = [(0, start_node_index)]
    
    while queue:
        dist, node = heapq.heappop(queue)
        
        # 最初のノードだけ避ける and 訪問済みはスルー
        if dist > distances[node] or visited[node]:
            continue
        
        visited[node] = True

        for neighbor, weight in graph[node]:
            new_distance = dist + weight
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))
                # np.append(converted_tab,note,axis=0)
    
    return distances

def estimate_tab_from_pred(tab):
    num_strings = 6
    num_frets = 21

    # graph = [[] for _ in range(num_strings * num_frets)]
    graph = np.empty([0,0,0])
    # print(graph)

    start_node_index = get_start_node_index(tab)
    prev_fret = start_node_index

    # 同音をノードに追加する
    for note in tab:
        pass

    for string in range(num_strings):
        for fret in range(num_frets):
            node = string * num_frets + fret
            for next_string in range(num_strings):
                for next_fret in range(num_frets):
                    next_node = next_string * num_frets + next_fret
                    # distance = abs(fret - next_fret)
                    distance = abs(prev_fret - (max(g) - min(g) /2)) + (max(g) - min(g))
                    if max(g) > 7:
                        distance += 1
                    graph[node].append((next_node, distance))

    distances = dijkstra(graph,start_node_index)

    estimated_tab = []
    # for node in range(num_strings * num_frets):
    #     string = node // num_frets
    #     fret = node % num_frets
    #     distance = distances[node]
    #     estimated_tab.append((string, fret, distance))

    return estimated_tab




def main():
    parser = argparse.ArgumentParser(description='code for plotting results')
    parser.add_argument("model", type=str,
                        help="name of trained model: ex) 202201010000")
    parser.add_argument("epoch", type=int,
                        help="number of model epoch to use: ex) 64")
    parser.add_argument("-v", "--verbose", help="option for verbosity: -v to turn on verbosity",
                        action="store_true", required=False, default=False)
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
        "input_feature_type": input_feature_type
    }

    if mode == "F0":
        npz_dir = os.path.join(
            "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz")
    elif mode == "tab":
        npz_dir = os.path.join(
            "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz")

    for test_num in range(6):
        if mode == "F0":
            visualize_dir = os.path.join(
                "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "visualize", f"test_0{test_num}")
        if mode == "tab":
            visualize_dir = os.path.join(
                "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "visualize", f"test_0{test_num}")

        npz_filename_list = glob.glob(
            os.path.join(npz_dir, f"test_0{test_num}", "*"))
        kwargs["visualize_dir"] = visualize_dir
        # if not(os.path.exists(visualize_dir)):
        #     os.makedirs(visualize_dir)
            
        # # paralell process
        # p = Pool(n_cores)
        # p.starmap(visualize, zip(npz_filename_list, repeat(kwargs)))
        # p.close()  # or p.terminate()
        # p.join()


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description='code for plotting results')
    parser.add_argument("model", type=str,
                        help="name of trained model: ex) 202201010000")
    parser.add_argument("epoch", type=int,
                        help="number of model epoch to use: ex) 64")
    parser.add_argument("-v", "--verbose", help="option for verbosity: -v to turn on verbosity",
                        action="store_true", required=False, default=False)
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    npz_dir = os.path.join(
        "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz")
    # npz_filename_list = glob.glob(
    #         os.path.join(npz_dir, f"test_0{test_num}", "*"))
    npz_filename_list = glob.glob(
        os.path.join(npz_dir, "test_00", "*"))
    
    npz_data = np.load(npz_filename_list[3])
    # print(npz_filename_list[3])
    note_pred = npz_data["note_tab_pred"]
    # print(len(note_pred))
    # print(note_pred[3].shape)
    print(note_pred.shape)
    estimated_tab = estimate_tab_from_pred(note_pred)

