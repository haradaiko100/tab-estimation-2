import os
import argparse
import numpy as np
from predict import tab2pitch
from const import GUITAR_SOUND_MATRICS


# 関数の名前ひどいから後で変える
def get_issue_data(tab, pred_tab):
    copied_tab = np.copy(tab)
    copied_pred_tab = np.copy(pred_tab)

    # 教師データと合っている正しい弦とフレットの組み合わせを削除する
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
        # 残った音の中で、鳴っている弦とフレットのペアを算出する
        sounding_pairs = {}
        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)
            if fret_position != 20:
                sound_on_string_fret_pairs = get_finger_positions_on_specific_sound(
                    fret_position, string_index
                )

                for string, fret in sound_on_string_fret_pairs.items():
                    if string in sounding_pairs:
                        sounding_pairs[string].append(fret)
                    else:
                        # 配列にキーが存在しない場合、新しいキーを作成して値を追加
                        sounding_pairs[string] = [fret]

        # 各弦の押している位置を取得(教師データの方)
        # finger_positions = np.argmax(note, axis=1)

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

        # 異弦同音となる音を取得する
        # for string_index, sound_on_specific_string_list in enumerate(note):
        #     fret_position = np.argmax(sound_on_specific_string_list)

        #     # ミュートの場合はスキップ

        #     same_sound_on_different_strings = get_finger_positions_on_specific_sound(
        #         fret=fret_position, string=string_index
        #     )

        #     # 予測されたタブ譜において、同じ音が鳴っている箇所を取得する
        #     pred_tab_note = copied_pred_tab[note_index]

        #     # 異弦同音と関連研究の出力が一致していたら、その音を保存する
        #     is_same_sound_on_different_strings = get_specific_pair_exits_or_not(
        #         same_sound_on_different_strings, string_index, fret_position
        #     )

        #     if is_same_sound_on_different_strings:
        #         # 音を保存する処理
        #         pass


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
