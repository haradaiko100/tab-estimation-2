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



def get_fingers_distance_dict(note:np.ndarray):

    # 各弦の押している位置を取得
    finger_positions = np.argmax(note,axis=1)

    # finger_positionsから開放弦と鳴らしてない弦の情報を削除
    new_finger_positions = [elem for elem in finger_positions if elem != 0 and elem != 21]

    # 要素ない場合は0を返す
    if len(new_finger_positions) == 0:
        return {
            "max":0,
            "min":0,
        }
    
    max_index = max(new_finger_positions)
    min_index = min(new_finger_positions)

    return {
        "max":max_index,
        "min":min_index,
    }

# カウントの情報とdataのnumpy配列を返す
def get_same_time_nodes_from_graph(graph,target_time):
    # same_time_nodes = np.empty([0,0,0])
    same_time_nodes_info = []
    for node in graph.nodes():
        if 'time' in graph.nodes[node] and graph.nodes[node]['time'] == target_time:
            node_chord_info = graph.nodes[node]["data"]
            
            same_time_nodes_info.append({
                "data":node_chord_info,
                "count":node,
                })

            # same_time_nodes.append(node_chord_info)
            # np.append(same_time_nodes,node_chord_info,axis=0)

    return same_time_nodes_info


def calc_weight_between_notes(prev_note:np.ndarray,current_note:np.ndarray):

    current_fingers_dict = get_fingers_distance_dict(current_note)
    prev_fingers_dict = get_fingers_distance_dict(prev_note)

    current_fingers_distance = current_fingers_dict["max"] - current_fingers_dict["min"]

    prev_fret =  sum(prev_fingers_dict.values()) / len(prev_fingers_dict)
    weight = abs(prev_fret - (current_fingers_distance /2)) + (current_fingers_distance)
    if max(current_fingers_dict["max"]) > 7:
        weight += 1

    return weight


def estimate_tab_from_pred(tab:np.ndarray):
    DG = nx.DiGraph()
    current_time = 0
    prev_time = 0
    node_count = 0

    for note in tab:
        
        if current_time == 0:
            DG.add_node(node_count,data=note,time=current_time,count=node_count)
            node_count += 1

        else:
            # get_same_note_nodes内でありえないノードは含まれないようにしたい
            same_note_nodes = get_same_note_nodes(note) # numpyの配列
            prev_time_nodes = get_same_time_nodes_from_graph(DG,prev_time)

            # グラフにノード追加
            for i in range(len(same_note_nodes)):
                DG.add_node(node_count,data=same_note_nodes[i],time=current_time,count=node_count)

                # エッジ追加
                for current_node in same_note_nodes:
                    for prev_node in prev_time_nodes:
                        weight = calc_weight_between_notes(current_note=current_node,prev_note=prev_node["data"])
                        DG.add_edge(node_count,prev_node["count"],weight=weight)
                
                # エッジ追加後にインクリメント
                node_count += 1


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