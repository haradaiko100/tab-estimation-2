import unittest
from dag import get_fingers_distance_dict
from move import get_same_note_nodes
import numpy as np
import networkx as nx

# この部分にテスト対象の関数とクラスをimportする記述が必要です


class TestDijkstraAlgorithm(unittest.TestCase):
    def test_dijkstra_algorithm(self):
        # Create a directed graph
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(1, 2, 2), (1, 3, 4), (2, 3, 1), (2, 4, 7), (3, 4, 3)]
        )

        # Calculate shortest path
        shortest_path = nx.dijkstra_path(G, source=1, target=4, weight="weight")
        expected_path = [1, 2, 3, 4]

        self.assertEqual(shortest_path, expected_path)

    def test_dijkstra_algorithm_different_graph(self):
        # Create another directed graph
        G = nx.DiGraph()
        G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (2, 4, 3), (3, 4, 1)])

        # Calculate shortest path
        shortest_path = nx.dijkstra_path(G, source=1, target=4, weight="weight")
        expected_path = [1, 3, 4]

        self.assertEqual(shortest_path, expected_path)

    def test_get_fingers_distance_dict_multiple_strings(self):
        note_A = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]

        fingers_dict_result = get_fingers_distance_dict(note_A)
        expected_dict = {"max": 8, "min": 7}

        self.assertEqual(fingers_dict_result, expected_dict)

    def test_get_fingers_distance_dict_single_string(self):
        note_A = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]

        fingers_dict_result = get_fingers_distance_dict(note_A)
        expected_dict = {"max": 7, "min": 7}

        self.assertEqual(fingers_dict_result, expected_dict)

    # def test_dijkstra_algorithm_on_tablature_graph(self):

    #     DG = nx.DiGraph()

    #     # same_note_nodes = get_same_note_nodes(note)

    #     # グラフにノード追加
    #     for i in range(len(same_note_nodes)):
    #         DG.add_node(
    #             node_count,
    #             data=same_note_nodes[i],
    #             time=current_time,
    #             count=node_count,
    #         )

    #         # エッジ追加
    #         for current_node in same_note_nodes:
    #             for prev_node in prev_time_nodes:
    #                 weight = calc_weight_between_notes(
    #                     current_note=current_node, prev_note=prev_node["data"]
    #                 )
    #                 DG.add_edge(prev_node["count"], node_count, weight=weight)

    #     # shortest_path = nx.dijkstra_path(
    #     #     G=DG, source=1, target=node_count - 1, weight="weight"
    #     # )
    #     shortest_path = nx.dijkstra_path(
    #         G=DG, source=1, target=10, weight="weight"
    #     )


if __name__ == "__main__":
    unittest.main()
