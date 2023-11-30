import os
import argparse
import numpy as np
from predict import tab2pitch
from const import GUITAR_SOUND_MATRICS


def get_common_pairs_from_both_dicts(dict1, dict2):
    common_pairs = [
        (key, value)
        for key, value in dict1.items()
        if key in dict2 and dict2[key] == value
    ]
    return common_pairs


def save_same_sound_issue_data(tab, pred_tab):
    same_sound_issue_data_dict = get_same_sound_issue_data(tab, pred_tab)

    tab_data_in_npz = np.array(same_sound_issue_data_dict["tab"])
    pred_data_in_npz = np.array(same_sound_issue_data_dict["pred"])


# この関数の引数は、教師データと出力されたデータのタブ譜だけ
# この関数を2回使うことで、関連研究の出力と自分のシステムの方の出力を比較する
def get_same_sound_issue_data(tab, pred_tab):
    tab_same_sound_issue_data_list = []
    pred_same_sound_issue_data_list = []

    all_muted_note = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    copied_tab = np.copy(tab)
    copied_pred_tab = np.copy(pred_tab)

    # 正しい弦とフレットの組み合わせを削除する（教師データと出力されたデータから）
    for note_index, note in enumerate(copied_tab):
        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)
            pred_tab_note_fret_position = np.argmax(
                copied_pred_tab[note_index][string_index]
            )

            if fret_position != 20 and fret_position == pred_tab_note_fret_position:
                # 元々鳴っていた音を削除
                sound_on_specific_string_list[fret_position] = 0
                copied_tab[note_index][string_index][pred_tab_note_fret_position] = 0

                # ミュートしている音として扱うようにする
                sound_on_specific_string_list[-1] = 1
                copied_tab[note_index][string_index][-1] = 1

    for note_index, note in enumerate(copied_tab):
        tab_specific_issue_data = np.zeros((6, 21))  # 教師データ
        pred_specific_issue_data = np.zeros((6, 21))  # 出力の方のデータ

        pred_fingers_positions = np.argmax(copied_pred_tab[note_index], axis=1)
        pred_fingers_positions = [
            elem for elem in pred_fingers_positions
        ]  # numpy配列から鳴っているフレットの情報を取得

        pred_fingers_dict = {
            index: value for index, value in enumerate(pred_fingers_positions)
        }

        # ミュート(インデックスが20)の弦の要素を辞書から削除
        pred_sounding_fingers_dict = {
            key: value for key, value in pred_fingers_dict.items() if value != 20
        }

        print("{0}番目: {1}".format(note_index, pred_sounding_fingers_dict))

        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)

            # 各弦で教師データで残った音から、異弦同音を算出
            if fret_position == 20:
                tab_specific_issue_data[string_index][-1] = 1
                pred_specific_issue_data[string_index][-1] = 1
                continue

            else:
                same_sound_string_fret_pairs = get_finger_positions_on_specific_sound(
                    string=string_index, fret=fret_position
                )

                print("same: ",same_sound_string_fret_pairs)
                common_string_fret_pairs = get_common_pairs_from_both_dicts(
                    same_sound_string_fret_pairs, pred_sounding_fingers_dict
                )

                print("common: ",common_string_fret_pairs)

                # 異弦同音だったとき
                if common_string_fret_pairs:
                    pred_issue_string = common_string_fret_pairs[0][0]
                    pred_issue_fret = common_string_fret_pairs[0][1]

                    tab_specific_issue_data[string_index][fret_position] = 1  # 教師データ
                    pred_specific_issue_data[pred_issue_string][
                        pred_issue_fret
                    ] = 1  # 出力の方のデータ

                else:
                    # 単純に音の高さが違う場合は、ミュートとして修正
                    tab_specific_issue_data[string_index][-1] = 1
                    pred_specific_issue_data[string_index][-1] = 1

        if not np.array_equal(pred_specific_issue_data, all_muted_note):
            tab_same_sound_issue_data_list.append(tab_specific_issue_data)
            pred_same_sound_issue_data_list.append(pred_specific_issue_data)

    return {
        "tab": tab_same_sound_issue_data_list,
        "pred": pred_same_sound_issue_data_list,
    }


def is_same_sound_issue(tab, pred_tab):
    copied_tab = np.copy(tab)
    copied_pred_tab = np.copy(pred_tab)

    # 正しい弦とフレットの組み合わせを削除する（教師データと出力されたデータから）
    for note_index, note in enumerate(copied_tab):
        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)
            pred_tab_note_fret_position = np.argmax(
                copied_pred_tab[note_index][string_index]
            )

            if fret_position != 20 and fret_position == pred_tab_note_fret_position:
                # 元々鳴っていた音を削除
                sound_on_specific_string_list[fret_position] = 0
                copied_tab[note_index][string_index][pred_tab_note_fret_position] = 0

                # ミュートしている音として扱うようにする
                sound_on_specific_string_list[-1] = 1
                copied_tab[note_index][string_index][-1] = 1

    for note_index, note in enumerate(copied_tab):
        sounding_pairs = {}

        # 教師データで残った音から、異弦同音を算出する
        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)
            if fret_position != 20:
                same_sound_string_fret_pairs = get_finger_positions_on_specific_sound(
                    string=string_index, fret=fret_position
                )

                for string, fret in same_sound_string_fret_pairs.items():
                    if string in sounding_pairs:
                        sounding_pairs[string].append(fret)
                    else:
                        # 配列にキーが存在しない場合、新しいキーを作成して値を追加
                        sounding_pairs[string] = [fret]

        # 出力に含まれる弦とフレットの組み合わせが異弦同音と同じかどうかを判定
        for string_index, pred_sound_on_specific_string_list in enumerate(
            copied_pred_tab[note_index]
        ):
            fret_position = np.argmax(pred_sound_on_specific_string_list)

            # ミュート以外の場合について
            if fret_position != 20:
                is_same_sound_on_different_strings = is_value_in_list_of_key(
                    sounding_pairs,
                    string_index,
                    fret_position,
                )

                if is_same_sound_on_different_strings:
                    return True

    return False


def is_value_in_list_of_key(dictionary, target_key, target_value):
    if target_key in dictionary:
        value_array = dictionary[target_key]
        return target_value in value_array
    return False


# def get_specific_pair_exits_or_not(dictionary, target_key, target_value):
#     return target_key in dictionary and dictionary[target_key] == target_value


# 出力は{string: fret}の形式
# 形式を配列にしなかったのは、必ずしも要素数が6になるとは限らないため
def get_finger_positions_on_specific_sound(string, fret):
    same_sound_different_strings = {}

    if fret >= len(GUITAR_SOUND_MATRICS[string]):
        return same_sound_different_strings

    target_value = GUITAR_SOUND_MATRICS[string][fret]

    for row_index, row in enumerate(GUITAR_SOUND_MATRICS):
        for col_index, value in enumerate(row):
            if col_index != 20 and value == target_value:
                same_sound_different_strings[row_index] = col_index

    return same_sound_different_strings


def get_problem_files_name():
    parser = argparse.ArgumentParser(description="code for plotting results")
    parser.add_argument(
        "model", type=str, help="name of trained model: ex) 202201010000"
    )
    parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    parser.add_argument(
        "-v",
        "--verbose",
        help="option for verbosity: -v to turn on verbosity",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    verbose = args.verbose
    npz_dir = os.path.join(
        "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
    )


if __name__ == "__main__":
    # get_problem_files_name()
    get_finger_positions_on_specific_sound(1, 2)
