import numpy as np


def get_fingers_distance_dict(note: np.ndarray):
    # 各弦の押している位置を取得
    finger_positions = np.argmax(note, axis=1)

    # finger_positionsから開放弦(0)と鳴らしてない弦(21)の情報を削除
    new_finger_positions = [
        elem for elem in finger_positions if elem != 0 and elem != 20
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
        # if current_fingers_dict["max"] > 7:
        #     weight += 1

    return weight


if __name__ == "__main__":
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

    muted_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    result = get_fingers_distance_dict(input_array)
    print(result)
    muted = get_fingers_distance_dict(muted_array)
    print(muted)
