import unittest
import matplotlib.pyplot as plt
from dag import (
    calc_weight_between_notes,
    get_same_time_nodes_from_graph,
    remove_duplicates,
)
from move import get_same_note_nodes

import numpy as np
import networkx as nx


class TestDijkstraAlgorithm(unittest.TestCase):
    def test_calc_weight_between_notes(self):
        prev_note = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        next_note_A = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        next_note_B = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        weight_A = calc_weight_between_notes(prev_note, next_note_A)
        weight_B = calc_weight_between_notes(prev_note, next_note_B)
        print("weight_A: ", weight_A)
        print("weight_B: ", weight_B)

        weight_A_is_lower_than_B = weight_A < weight_B
        expected_result = True

        self.assertEqual(weight_A_is_lower_than_B, expected_result)

    def test_dag_E_to_E(self):
        tab = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
            ]
        )

        DG = nx.DiGraph()
        current_time = 0
        prev_time = 0
        node_count = 1
        dest_node_count = 1

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
                            current_note_being_numpy_list_shape = np.array(current_node)
                            prev_note_being_numpy_list_shape = np.array(
                                prev_node["data"]
                            )
                            weight = calc_weight_between_notes(
                                prev_note=prev_note_being_numpy_list_shape,
                                current_note=current_note_being_numpy_list_shape,
                            )
                            DG.add_edge(prev_node["count"], node_count, weight=weight)

                    # エッジ追加後にインクリメント
                    node_count += 1

            # 時間をインクリメント
            prev_time = current_time
            current_time += 1

        # 小節ごとに最短経路を求める
        # for i in range(len(stopping_point_node_list)):
        #     each_shortest_path = nx.dijkstra_path(
        #         G=DG,
        #         source=current_stopping_point_index,
        #         target=stopping_point_node_list[i],
        #         weight="weight",
        #     )
        #     # shortest_path_list.append(each_shortest_path)
        #     shortest_path_list += each_shortest_path

        #     current_stopping_point_index = stopping_point_node_list[i]

        shortest_path_list = nx.dijkstra_path(
            G=DG, source=1, target=dest_node_count, weight="weight"
        )

        # 重複あったら削除する
        shortest_path_list = remove_duplicates(shortest_path_list)

        # shortest_pathの実際のデータを取得する
        estimated_tab = [DG.nodes[node]["data"] for node in shortest_path_list]

        estimated_result_truth = True
        result = np.array_equal(tab, estimated_tab)

        # self.assertEqual(estimated_result_truth,result)

        for node, data in DG.nodes(data=True):
            print(f'Node {node}: {data["data"]}')

        # 各エッジの重みを出力
        for u, v, w in DG.edges(data=True):
            print(f'Edge ({u}, {v}) has weight: {w["weight"]}')

        print(estimated_tab)


if __name__ == "__main__":
    unittest.main()
