import networkx as nx
import numpy as np


def find_same_notes():

    a = [21, 21, 7, 7, 21, 7]
    results = [a]

    # インデックスごとの移動先と値の変動を定義
    movements = {
        0: [(1, -5)],
        1: [(2, -5)],
        2: [(3, -5), (1, +5)],
        3: [(4, -4), (2, +5)],
        4: [(5, -5), (3, +4)],
        5: [(4, +5)]
    }

    for i in range(len(a)):
        if a[i] != 21:  # 値が21でない要素に対してのみ並び替えを適用
            for move in movements[i]:
                next_index, value_change = move
                new_a = a.copy()  # 新しい配列を作成して並び替えを適用
                # 要素の位置を入れ替える
                new_a[i], new_a[next_index] = new_a[next_index], new_a[i]
                results.append(new_a)  # 並び替え結果を保存

    print(results)

    # 各要素を移動させて新しい配列を作成
    # result = []
    # for i in range(len(a)):
    #     new_value = a[i]
    #     for move in movements.get(i, []):
    #         next_index, value_change = move
    #         new_value += value_change
    #         a[next_index] += value_change
    #     result.append(new_value)

    # print(result)

def get_positions_with_ones(array:np.ndarray):
    positions_with_ones = []
    for row in array:
        positions = np.where(row == 1)[0]
        positions_with_ones.append(positions)
    return positions_with_ones

def get_note_in_npz(note):

    num_strings = 6  # 弦の数
    num_frets = 21  # フレットの数

    same_note_nodes = np.empty([0, 0, 0])

    # 21列目にある1の数を取得
    muted_strings_num = np.sum(note[:, 20])

    # なっている音の数
    sounding_strings_num = num_strings - muted_strings_num

    # 音のずらす数によってパターンを数え上げる
    for i in range(1, sounding_strings_num+1):

        for j in range(num_strings):
            if 


if __name__ == "__main__":

    # グラフを作成
    G = nx.DiGraph()

    # ノードを追加し、dataプロパティを設定
    G.add_node(1, data=np.array([1, 2, 3]))
    G.add_node(2, data=np.array([4, 5, 6]))
    G.add_node(3, data=np.array([7, 8, 9]))

    # エッジを追加
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)

    # ダイクストラ法で最短経路を求める
    shortest_path = nx.dijkstra_path(G, 1, 3)
    # print(shortest_path)
    print(G.nodes[1])
    for node in G.nodes():
        node_data = G.nodes[node]
        # time = node_data['time']
        data = node_data['data']
        # print(node)
        # print(data)
    # 経路に含まれるノードのdataプロパティを一つの配列にまとめる
    data_list = [G.nodes[node]['data'] for node in shortest_path]

    # 結果を出力
    # print(data_list)
    array = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    print(array[0][20])
    # 21列目にある1の数を取得
    ones_count = np.sum(array[:, 20])

    # print("21列目にある1の数:", ones_count)
