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

import networkx as nx


def get_same_note_nodes(note:np.ndarray) -> np.ndarray:
    pass



def get_fingers_distance(note:np.ndarray):

    # 各弦の押している位置を取得
    finger_positions = np.argmax(note,axis=1)

    # finger_positionsから開放弦と鳴らしてない弦の情報を削除
    new_finger_positions = [elem for elem in finger_positions if elem != 0 and elem != 21]

    # 要素ない場合は0を返す
    if len(new_finger_positions) == 0:
        return [0,0]
    
    max_index = max(new_finger_positions)
    min_index = min(new_finger_positions)

    return [max_index,min_index]


def get_same_time_nodes_from_graph(graph,target_time):
    same_time_nodes = []
    for node in graph.nodes():
        if 'time' in graph.nodes[node] and graph.nodes[node]['time'] == target_time:
            node_info = graph.nodes[node]
            same_time_nodes.append(node)

    return same_time_nodes


def calc_weight_between_notes(prev_note:np.ndarray,current_note:np.ndarray):

    current_depressing_fingers_dist = get_fingers_distance(current_note)
    prev_depressing_fingers_dist = get_fingers_distance(prev_note)

    prev_fret =  sum(prev_depressing_fingers_dist) / len(prev_depressing_fingers_dist)
    weight = abs(prev_fret - (current_depressing_fingers_dist /2)) + (current_depressing_fingers_dist)
    if max(current_depressing_fingers_dist) > 7:
        weight += 1

    return weight


def estimate_tab_from_pred(tab:np.ndarray):
    DG = nx.DiGraph()
    graph = np.empty([0,0,0])
    current_time = 0
    prev_time = 0
    node_count = 0

    # 現時刻の音について
    for note in tab:

        # get_same_note_nodes内でありえないノードは含まれないようにしたい
        same_note_nodes = get_same_note_nodes(note)

        # for node in same_note_nodes:
        #     DG.add_node(node_count, data=data,time=current_time)
        #     node_count += 1

        # グラフにノード追加
        for i in range(len(same_note_nodes)):
            DG.add_node(node_count,data=same_note_nodes[i],time=current_time,count=node_count)
            node_count += 1

        # エッジ追加
        if current_time != 0:
            prev_time_nodes = get_same_time_nodes_from_graph(DG,prev_time)

            for current_node in same_note_nodes:
                for prev_node in prev_time_nodes:
                    weight = calc_weight_between_notes(current_note=current_node,prev_note=prev_node)
                    DG.add_edge(DG.nodes[prev_node]["count"],DG.nodes[current_node]["count"],weight=weight)

        # 時間をインクリメント
        prev_time = current_time
        current_time += 1


    shortest_path = nx.dijkstra_path(DG, 0, node_count,weight="weight")

    # shortest_pathの実際のデータを取得する
    note_list = [DG.nodes[node]['data'] for node in shortest_path]


    return note_list


if __name__ == "__main__":
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