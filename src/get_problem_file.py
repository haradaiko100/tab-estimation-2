import os
import argparse
import numpy as np
from predict import tab2pitch
from const import GUITAR_SOUND_MATRICS


# この関数の引数は、教師データと出力されたデータのタブ譜だけ
# この関数を2回使うことで、関連研究の出力と自分のシステムの方の出力を比較する
def save_same_sound_on_different_strings(tab, pred_tab):
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
                    fret_position, string_index
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
                    # 音を保存する処理
                    pass


def is_value_in_list_of_key(dictionary, target_key, target_value):
    if target_key in dictionary:
        value_array = dictionary[target_key]
        return target_value in value_array
    return False


# def get_specific_pair_exits_or_not(dictionary, target_key, target_value):
#     return target_key in dictionary and dictionary[target_key] == target_value


# 出力は{string: fret}の形式
# 形式を配列にしなかったのは、必ずしも要素数が6になるとは限らないため
def get_finger_positions_on_specific_sound(fret, string):
    same_sound_different_strings = {}

    if fret >= len(GUITAR_SOUND_MATRICS[string]):
        return same_sound_different_strings

    target_value = GUITAR_SOUND_MATRICS[string][fret]

    for row_index, row in enumerate(GUITAR_SOUND_MATRICS):
        for col_index, value in enumerate(row):
            if value == target_value:
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
