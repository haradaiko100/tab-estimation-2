import unittest
import numpy as np
import networkx as nx

# この部分にテスト対象の関数とクラスをimportする記述が必要です

class TestDijkstraAlgorithm(unittest.TestCase):
    def test_dijkstra_algorithm(self):
        # Create a directed graph
        G = nx.DiGraph()
        G.add_weighted_edges_from([(1, 2, 2), (1, 3, 4), (2, 3, 1), (2, 4, 7), (3, 4, 3)])
        
        # Calculate shortest path
        shortest_path = nx.dijkstra_path(G, source=1, target=4, weight='weight')
        expected_path = [1, 2, 3, 4]
        
        self.assertEqual(shortest_path, expected_path)

    def test_dijkstra_algorithm_different_graph(self):
        # Create another directed graph
        G = nx.DiGraph()
        G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (2, 4, 3), (3, 4, 1)])
        
        # Calculate shortest path
        shortest_path = nx.dijkstra_path(G, source=1, target=4, weight='weight')
        expected_path = [1, 3, 4]
        
        self.assertEqual(shortest_path, expected_path)


if __name__ == "__main__":
    unittest.main()
