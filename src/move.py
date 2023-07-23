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
        5: [(4, +5)],
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


def shift_positions(array):
    shifted_array = np.zeros_like(array)
    for i in range(array.shape[0] - 1):
        row = array[i]
        # print("${0} : ${1}".format(i,row))
        next_row = array[i + 1]
        if row[-1] == 1:
            continue

        # 一つ下の行の1の位置をずらし、列を左に5ずらす
        positions = np.where(row == 1)[0]
        shifted_positions = positions + 5
        shifted_positions[shifted_positions >= array.shape[1] - 1] = array.shape[1] - 1

        # 一つ下の行の対応する位置に1を配置
        next_row[shifted_positions] = 1
        shifted_array[i + 1] = next_row

    # 各行について1が複数個ある場合、インデックスが20が含まれない場合、ずらす操作をする前にあった1について(a)を行う
    for i in range(array.shape[0]):
        row = array[i]
        if np.sum(row) > 1 and row[-1] != 1:
            positions = np.where(row == 1)[0]
            shifted_positions = positions + 5
            shifted_positions[shifted_positions >= array.shape[1] - 1] = (
                array.shape[1] - 1
            )
            row[shifted_positions] = 0

    return shifted_array


def get_same_note_with_one_move(note):
    num_strings = 6
    same_note_list = []

    # downwards_shift
    for start_string in range(num_strings - 1):
        copied_array = np.copy(note)
        sounding_position = np.where(copied_array[start_string] == 1)[0]
        # print(sounding_position)

        # 弦がミュートしている場合はスキップ
        if len(sounding_position) == 1 and copied_array[start_string][-1] == 1:
            continue

        # 3→2弦の移動は-4、それ以外は-5
        parallel_move_dist = 4 if start_string == 3 else 5

        # フレットがマイナスだったらスキップ
        next_string_fret = sounding_position[0] - parallel_move_dist
        if next_string_fret < 0:
            continue

        # 一つ下の弦に音を移動させる
        copied_value = np.copy(copied_array[start_string, sounding_position[0]])
        copied_array[start_string + 1][next_string_fret] = copied_value

        # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
        if copied_array[start_string + 1][-1] == 1:
            copied_array[start_string + 1][-1] = 0

        # 移動元の弦の状態を更新する
        copied_array[start_string][sounding_position[0]] = 0
        copied_array[start_string][-1] = 1

        moved_string_position = np.where(copied_array[start_string + 1] == 1)[0]

        # 移動先の弦のなっている箇所が1つだけだったら追加する
        if len(moved_string_position) == 1:
            same_note_list.append(copied_array)

    # upwards_shift
    for start_string in range(num_strings - 1):
        reversed_copied_array = np.copy(note)[::-1]
        sounding_position = np.where(reversed_copied_array[start_string] == 1)[0]
        # print(sounding_position)

        # 弦がミュートしている場合はスキップ
        if len(sounding_position) == 1 and reversed_copied_array[start_string][-1] == 1:
            continue

        # 2→3弦の移動は+4、それ以外は+5
        parallel_move_dist = 4 if start_string == 1 else 5

        # フレットが20以上だったらスキップ
        next_string_fret = sounding_position[0] + parallel_move_dist
        if next_string_fret >= 20:
            continue

        # 一つ下の弦に音を移動させる
        copied_value = np.copy(
            reversed_copied_array[start_string, sounding_position[0]]
        )

        reversed_copied_array[start_string + 1][next_string_fret] = copied_value

        # 移動先の弦が元々ミュートしていた場合、そのフレット情報を更新する
        if reversed_copied_array[start_string + 1][-1] == 1:
            reversed_copied_array[start_string + 1][-1] = 0

        # 移動元の弦の状態を更新する
        reversed_copied_array[start_string][sounding_position[0]] = 0
        reversed_copied_array[start_string][-1] = 1

        moved_string_position = np.where(reversed_copied_array[start_string + 1] == 1)[
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
            # print(sounding_position)
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


def get_positions_with_ones(array: np.ndarray):
    positions_with_ones = []
    for row in array:
        positions = np.where(row == 1)[0]
        positions_with_ones.append(positions)
    return positions_with_ones


def get_same_note_in_npz(note):
    num_strings = 6  # 弦の数
    num_frets = 21  # フレットの数

    same_note_nodes = np.empty([0, 0, 0])

    # 21列目にある1の数を取得
    muted_strings_num = np.sum(note[:, 20])

    # 鳴っている音の数
    sounding_strings_num = num_strings - muted_strings_num

    # 音が鳴っていないとき or 全ての弦が鳴っているとき
    if sounding_strings_num == 0 or sounding_strings_num == num_strings:
        return note

    # 音のずらす数によってパターンを数え上げる
    for shift_position_num in range(1, sounding_strings_num + 1):
        # start_guitar_stringは移動を最初にする弦
        for start_guitar_string in range(num_strings):
            sounding_position = np.where(note[start_guitar_string] == 1)[0]

            # ミュートしているだけの場合はスキップ
            if len(sounding_position) == 1 and note[start_guitar_string][-1] == 1:
                continue

            # まずはdownwards_shiftを実装していく

            note[start_guitar_string + 1][sounding_position - 5] = note[
                start_guitar_string
            ][sounding_position]

            note[start_guitar_string][sounding_position] = 0


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

    # 結果を出力
    # print(data_list)
    array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    result = get_positions_with_ones(array)
    print(result)
    # 21列目にある1の数を取得
    ones_count = np.sum(array[:, 20])

    input_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    print(input_array)

    same_note = get_same_note_with_one_move(input_array)
    # print(same_note)
    same_note_positions = get_same_note_positions(input_array)
    print(same_note_positions)

    # test_result = np.where(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]) == 1)[0]
    # print(test_result)
    # test_array = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    # print(np.any(test_array == 1))

    # print(input_array[1]) # input_array[0] → 0行目
    # result_array = shift_positions(input_array)
    # print(result_array)

    # print("21列目にある1の数:", ones_count)
