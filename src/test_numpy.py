import numpy as np


def get_sounding_string_fret_pairs_dict(note):
    finger_positions = np.argmax(note, axis=1)

    # numpy配列から鳴っているフレットの情報を取得
    sounding_finger_positions = [elem for elem in finger_positions]

    print(sounding_finger_positions)


def common_elements(dict1, dict2):
    common_items = [
        (key, value)
        for key, value in dict1.items()
        if key in dict2 and dict2[key] == value
    ]
    return common_items


def main():
    # 例として3x3の行列を作成
    array_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 特定の要素の値を変更（例: 2行2列の要素を100に変更）
    row_index = 1  # 行のインデックス（0から始まるインデックス）
    column_index = 1  # 列のインデックス（0から始まるインデックス）
    new_value = 100

    array_3x3[row_index][column_index] = new_value

    # 結果の表示（必要であれば）
    print(array_3x3)


if __name__ == "__main__":
    # main()
    # next_note_B = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     ]
    # )
    # get_sounding_string_fret_pairs_dict(next_note_B)
    # サンプルの辞書を作成
    dict1 = {0: 1, 1: 2, 2: 3, "d": 4}
    dict2 = {0: 1, 1: 4, 2: 5, "e": 6}

    # 共通の要素を取得
    result = common_elements(dict1, dict2)

    print(result[0][1])
    # 結果の表示
    print(result)
