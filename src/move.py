import networkx as nx
import numpy as np


def get_same_note_with_one_move(note):
    num_strings = 6
    same_note_list = []

    # downwards_shift
    for start_string in range(1, num_strings):
        copied_array = np.copy(note)
        sounding_position = np.where(copied_array[start_string] == 1)[0]
        # print(sounding_position)

        # 弦がミュートしている場合はスキップ
        if len(sounding_position) == 1 and copied_array[start_string][-1] == 1:
            continue

        # 3→2弦の移動は-4、それ以外は-5
        parallel_move_dist = 4 if start_string == 2 else 5

        # フレットがマイナスだったらスキップ
        next_string_fret = sounding_position[0] - parallel_move_dist
        if next_string_fret < 0:
            continue

        # 一つ下の弦に音を移動させる
        copied_value = np.copy(copied_array[start_string, sounding_position[0]])
        copied_array[start_string - 1][next_string_fret] = copied_value

        # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
        if copied_array[start_string - 1][-1] == 1:
            copied_array[start_string - 1][-1] = 0

        # 移動元の弦の状態を更新する
        copied_array[start_string][sounding_position[0]] = 0
        copied_array[start_string][-1] = 1

        moved_string_position = np.where(copied_array[start_string - 1] == 1)[0]

        # 移動先の弦のなっている箇所が1つだけだったら追加する
        if len(moved_string_position) == 1:
            same_note_list.append(copied_array)

    # upwards_shift
    for start_string in range(1, num_strings):
        reversed_copied_array = np.copy(note)[::-1]
        sounding_position = np.where(reversed_copied_array[start_string] == 1)[0]
        # print(sounding_position)

        # 弦がミュートしている場合はスキップ
        if len(sounding_position) == 1 and reversed_copied_array[start_string][-1] == 1:
            continue

        # 2→3弦の移動は+4、それ以外は+5
        parallel_move_dist = 4 if start_string == 4 else 5

        # フレットが20以上だったらスキップ
        next_string_fret = sounding_position[0] + parallel_move_dist
        if next_string_fret >= 20:
            continue

        # 一つ下の弦に音を移動させる
        copied_value = np.copy(
            reversed_copied_array[start_string, sounding_position[0]]
        )

        reversed_copied_array[start_string - 1][next_string_fret] = copied_value

        # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
        if reversed_copied_array[start_string - 1][-1] == 1:
            reversed_copied_array[start_string - 1][-1] = 0

        # 移動元の弦の状態を更新する
        reversed_copied_array[start_string][sounding_position[0]] = 0
        reversed_copied_array[start_string][-1] = 1

        moved_string_position = np.where(reversed_copied_array[start_string - 1] == 1)[
            0
        ]

        # 移動先の弦のなっている箇所が1つだけだったら追加する
        if len(moved_string_position) == 1:
            # 逆順にして元に戻す
            same_note_list.append(reversed_copied_array[::-1])

    return same_note_list


# 複数の弦を移動させる必要がある場合
def get_same_note_positions(note):
    num_strings = 6
    same_note_list = []

    # downwards_shift
    if note[num_strings - 1][-1] == 1:
        copied_array = np.copy(note)
        can_sound_correctly = True
        for string in range(num_strings - 1):
            sounding_position = np.where(copied_array[string] == 1)[0]
            # print(sounding_position)
            # 弦がミュートしている場合はスキップ
            if len(sounding_position) == 1 and copied_array[string][-1] == 1:
                continue

            # 元々の弦がミュートしている場合はスキップ(2回以上同じ音に対して移動させないため)
            if note[string][-1] == 1:
                continue

            # 3→2弦の移動は-4、それ以外は-5
            parallel_move_dist = 4 if string == 3 else 5

            prev_sounding_fret = max(sounding_position)

            # フレットがマイナスだったらそのパターンは無し
            next_string_fret = prev_sounding_fret - parallel_move_dist
            if next_string_fret < 0:
                can_sound_correctly = False
                break

            # 一つ下の弦に音を移動させる
            copied_value = np.copy(copied_array[string, prev_sounding_fret])
            copied_array[string + 1][next_string_fret] = copied_value

            # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
            if copied_array[string + 1][-1] == 1:
                copied_array[string + 1][-1] = 0

            # 移動元の弦の状態を更新する
            copied_array[string][prev_sounding_fret] = 0

            # 移動元の弦の状態を更新後、何も音の情報がない場合(移動先の弦が元々ミュートしていた場合)
            if not np.any(copied_array[string] == 1):
                copied_array[string][-1] = 1

        if can_sound_correctly:
            same_note_list.append(copied_array)

    # upwards_shift
    if note[0][-1] == 1:
        reversed_copied_array = np.copy(note)[::-1]
        can_sound_correctly = True
        for string in range(num_strings - 1):
            sounding_position = np.where(reversed_copied_array[string] == 1)[0]

            # 弦がミュートしている場合はスキップ
            if len(sounding_position) == 1 and reversed_copied_array[string][-1] == 1:
                continue

            # 元々の弦がミュートしている場合はスキップ(2回以上同じ音に対して移動させないため)
            if note[num_strings - 1 - string][-1] == 1:
                continue

            # 2→3弦の移動は+4、それ以外は+5
            parallel_move_dist = 4 if string == 1 else 5

            prev_sounding_fret = min(sounding_position)

            # フレットがギターの幅超えてたらそのパターンは無し
            next_string_fret = prev_sounding_fret + parallel_move_dist
            if next_string_fret >= 20:
                can_sound_correctly = False
                break

            # 一つ下の弦に音を移動させる
            copied_value = np.copy(reversed_copied_array[string, prev_sounding_fret])
            reversed_copied_array[string + 1][next_string_fret] = copied_value

            # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
            if reversed_copied_array[string + 1][-1] == 1:
                reversed_copied_array[string + 1][-1] = 0

            # 移動元の弦の状態を更新する
            reversed_copied_array[string][prev_sounding_fret] = 0

            # 移動元の弦の状態を更新後、何も音の情報がない場合(移動先の弦が元々ミュートしていた場合)
            if not np.any(reversed_copied_array[string] == 1):
                reversed_copied_array[string][-1] = 1

        if can_sound_correctly:
            # 逆順にして戻す
            same_note_list.append(reversed_copied_array[::-1])

    return same_note_list


def get_same_note_nodes(note):
    num_strings = 6  # 弦の数

    muted_strings_num = np.sum(note[:, -1])
    sounding_strings_num = num_strings - muted_strings_num

    same_note_nodes = []
    same_note_nodes.append(note)

    if sounding_strings_num == 1:
        same_note_with_one_move = get_same_note_with_one_move(note)
        for i in range(len(same_note_with_one_move)):
            same_note_nodes.append(same_note_with_one_move[i])

    elif sounding_strings_num >= 2:
        same_note_with_one_move = get_same_note_with_one_move(note)
        for i in range(len(same_note_with_one_move)):
            same_note_nodes.append(same_note_with_one_move[i])

        same_note_positions = get_same_note_positions(note)
        for i in range(len(same_note_positions)):
            same_note_nodes.append(same_note_positions[i])

    return same_note_nodes


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
    # print(G.nodes[1])
    for node in G.nodes():
        node_data = G.nodes[node]
        # time = node_data['time']
        data = node_data["data"]
        # print(node)
        # print(data)
    # 経路に含まれるノードのdataプロパティを一つの配列にまとめる
    data_list = [G.nodes[node]["data"] for node in shortest_path]
    print(data_list)
    # print(type(data_list).__module__ == "numpy")
    npz_data_list = np.array(data_list)
    # print(type(npz_data_list).__module__ == "numpy")

    input_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    result = get_same_note_nodes(input_array)
    print(result)
    # print(len(result))
