import os
import argparse
import numpy as np
from predict import tab2pitch
import pandas as pd
import glob
from datetime import datetime
from const import GUITAR_SOUND_MATRICS


def get_common_pairs_from_both_dicts(dict1, dict2):
    common_pairs = [
        (key, value)
        for key, value in dict1.items()
        if key in dict2 and dict2[key] == value
    ]
    return common_pairs


def save_same_same_sound_issue_data_in_npz(
    existing_npz_path,
    new_npz_file_path,
    mode,
):
    # 既存のnpzファイルからデータを読み込む
    existing_data = np.load(existing_npz_path)
    print("existing_npz_path: ", existing_npz_path)

    # 既存のデータを取得
    existing_data_dict = {key: existing_data[key] for key in existing_data.files}
    tab = existing_data["note_tab_gt"]

    if mode == "pred_tab":
        pred_tab = existing_data["note_tab_pred"]
        same_sound_issue_data_dict = get_same_sound_issue_data_from_tab_and_CNN(
            tab, pred_tab
        )

    elif mode == "graph_tab":
        graph_tab = existing_data["note_tab_graph_pred"]
        same_sound_issue_data_dict = get_same_sound_issue_data_from_tab_and_graph(
            tab, graph_tab
        )

    else:
        print("mode is invalid")
        return

    # 新しいデータを既存のデータに結合または追加
    for key, value in same_sound_issue_data_dict.items():
        if key not in existing_data_dict:
            existing_data_dict[key] = value

    # 新しいデータを含む辞書を作成
    new_data = {**existing_data_dict}

    # NPZファイルに保存
    np.savez(new_npz_file_path, **new_data)


def get_and_save_same_sound_issue_data(
    npz_filename_list, test_num, trained_model, use_model_epoch, date
):
    npz_filename_list = npz_filename_list.split("\n")

    npz_save_dir = os.path.join(
        "result",
        "same_sound_issue_data",
        f"{trained_model}_epoch{use_model_epoch}",
        date,
        "npz",
        f"test_0{test_num}",
    )

    if not (os.path.exists(npz_save_dir)):
        os.makedirs(npz_save_dir)

    # 教師データとCNNから異弦同音のみを抽出して保存
    for npz_file in npz_filename_list:
        npz_save_filename = os.path.join(npz_save_dir, os.path.split(npz_file)[1])

        # まず先に教師データとCNNの出力で異弦同音を抽出して保存
        save_same_same_sound_issue_data_in_npz(
            new_npz_file_path=npz_save_filename,
            existing_npz_path=npz_file,
            mode="pred_tab",
        )

        # その後に、教師データとグラフの出力で異弦同音を抽出して保存
        save_same_same_sound_issue_data_in_npz(
            new_npz_file_path=npz_save_filename,
            existing_npz_path=npz_file,
            mode="graph_tab",
        )

        print(f"finished {os.path.split(npz_file)[1][:-4]}")

    return


def get_same_sound_issue_data_from_tab_and_graph(tab, graph_tab):
    pred_same_sound_issue_data_list = []

    copied_tab = np.copy(tab)
    copied_graph_tab = np.copy(graph_tab)

    for note_index, note in enumerate(copied_tab):
        tab_specific_issue_data = np.zeros((6, 21))  # 教師データ
        pred_specific_issue_data = np.zeros((6, 21))  # グラフの方のデータ

        # 21列目の要素を1に変更
        tab_specific_issue_data[:, 20] = 1
        pred_specific_issue_data[:, 20] = 1

        pred_fingers_positions = np.argmax(copied_graph_tab[note_index], axis=1)
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

        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)

            # 各弦で教師データで残った音から、異弦同音を算出
            if fret_position != 20:
                same_sound_string_fret_pairs = get_finger_positions_on_specific_sound(
                    string=string_index, fret=fret_position
                )

                common_string_fret_pairs = get_common_pairs_from_both_dicts(
                    same_sound_string_fret_pairs, pred_sounding_fingers_dict
                )

                # 異弦同音だったとき
                if common_string_fret_pairs:
                    pred_issue_string = common_string_fret_pairs[0][0]
                    pred_issue_fret = common_string_fret_pairs[0][1]

                    tab_specific_issue_data[string_index][fret_position] = 1  # 教師データ
                    pred_specific_issue_data[pred_issue_string][
                        pred_issue_fret
                    ] = 1  # 出力の方のデータ

                    # ミュートとしている部分を修正
                    tab_specific_issue_data[string_index][-1] = 0
                    pred_specific_issue_data[pred_issue_string][-1] = 0

        pred_same_sound_issue_data_list.append(pred_specific_issue_data)

    pred_same_sound_issue_data_list_npz = np.array(pred_same_sound_issue_data_list)

    return {
        "graph_pred": pred_same_sound_issue_data_list_npz,
    }


# この関数の引数は、教師データと出力されたデータのタブ譜だけ
# この関数を2回使うことで、関連研究の出力と自分のシステムの方の出力を比較する
def get_same_sound_issue_data_from_tab_and_CNN(tab, pred_tab):
    tab_same_sound_issue_data_list = []
    pred_same_sound_issue_data_list = []

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

        # 21列目の要素を1に変更
        tab_specific_issue_data[:, 20] = 1
        pred_specific_issue_data[:, 20] = 1

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

        for string_index, sound_on_specific_string_list in enumerate(note):
            fret_position = np.argmax(sound_on_specific_string_list)

            # 各弦で教師データで残った音から、異弦同音を算出
            if fret_position != 20:
                same_sound_string_fret_pairs = get_finger_positions_on_specific_sound(
                    string=string_index, fret=fret_position
                )

                common_string_fret_pairs = get_common_pairs_from_both_dicts(
                    same_sound_string_fret_pairs, pred_sounding_fingers_dict
                )

                # 異弦同音だったとき
                if common_string_fret_pairs:
                    pred_issue_string = common_string_fret_pairs[0][0]
                    pred_issue_fret = common_string_fret_pairs[0][1]

                    tab_specific_issue_data[string_index][fret_position] = 1  # 教師データ
                    pred_specific_issue_data[pred_issue_string][
                        pred_issue_fret
                    ] = 1  # 出力の方のデータ

                    # ミュートとしている部分を修正
                    tab_specific_issue_data[string_index][-1] = 0
                    pred_specific_issue_data[pred_issue_string][-1] = 0

        tab_same_sound_issue_data_list.append(tab_specific_issue_data)
        pred_same_sound_issue_data_list.append(pred_specific_issue_data)

    tab_same_sound_issue_data_list_npz = np.array(tab_same_sound_issue_data_list)
    pred_same_sound_issue_data_list_npz = np.array(pred_same_sound_issue_data_list)

    return {
        "tab": tab_same_sound_issue_data_list_npz,
        "cnn_pred": pred_same_sound_issue_data_list_npz,
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


def main():
    parser = argparse.ArgumentParser(description="code for plotting results")
    parser.add_argument(
        "model", type=str, help="name of trained model: ex) 202201010000"
    )
    parser.add_argument("epoch", type=int, help="number of model epoch to use: ex) 64")
    parser.add_argument(
        "directory_date", type=str, help="date of dag.py was carried out"
    )  # dag.pyでできたディレクトリの日付を指定する
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    date = args.directory_date

    mode = "tab"
    n_cores = 12

    if mode == "F0":
        npz_dir = os.path.join(
            "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )
    elif mode == "tab":
        npz_dir = os.path.join(
            "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz"
        )

    result_path = os.path.join("result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for test_num in range(6):
        if mode == "tab":
            visualize_dir = os.path.join(
                "result",
                "same_sound_issue_data",
                f"{trained_model}_epoch{use_model_epoch}",
                date,
                "visualize",
                f"test_0{test_num}",
            )

        npz_filename_list = glob.glob(os.path.join(npz_dir, f"test_0{test_num}", "*"))
        if isinstance(npz_filename_list, list):
            npz_filename_list = "\n".join(npz_filename_list)
        # kwargs["visualize_dir"] = visualize_dir
        if not (os.path.exists(visualize_dir)):
            os.makedirs(visualize_dir)

        get_and_save_same_sound_issue_data(
            npz_filename_list=npz_filename_list,
            test_num=test_num,
            trained_model=trained_model,
            use_model_epoch=use_model_epoch,
            date=date,
        )

    return


if __name__ == "__main__":
    main()
