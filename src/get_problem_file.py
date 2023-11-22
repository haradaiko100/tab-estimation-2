import os
import argparse
import numpy as np
from predict import tab2pitch
from src.const.const import GUITAR_SOUND_MATRICS


# 関数の名前ひどいから後で変える
def get_issue_data(tab, pred_tab):
    # まず正解のタブ譜と予測されたタブ譜において、正解となる音は排除する必要がある
    # 排除するのは教師データの方だけで良い事に注意

    pass


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


def find_value_positions(matrix, target_value):
    positions = []

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if value == target_value:
                positions.append((row_index, col_index))

    return positions


# 例として、次のような二次元配列を使ってみます
# example_matrix = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 2, 11, 12],
#     [13, 14, 2, 16]
# ]

# target_value = 2

# result = find_value_positions(example_matrix, target_value)
# print(f"Target Value: {target_value}")
# print("Positions:", result)


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
