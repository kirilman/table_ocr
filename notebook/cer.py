import typing
import numpy as np
from itertools import groupby
from pathlib import Path
import os
import numpy as np
import pandas as pd
import argparse


def ctc_decoder(
    predictions: np.ndarray, chars: typing.Union[str, list]
) -> typing.List[str]:
    """CTC greedy decoder for predictions

    Args:
        predictions (np.ndarray): predictions from model
        chars (typing.Union[str, list]): list of characters

    Returns:
        typing.List[str]: list of words
    """
    # use argmax to find the index of the highest probability
    argmax_preds = np.argmax(predictions, axis=-1)

    # use groupby to find continuous same indexes
    grouped_preds = [[k for k, _ in groupby(preds)] for preds in argmax_preds]

    # convert indexes to chars
    texts = [
        "".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds
    ]

    return texts


def edit_distance(
    prediction_tokens: typing.List[str], reference_tokens: typing.List[str]
) -> int:
    """Standard dynamic programming algorithm to compute the Levenshtein Edit Distance Algorithm

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    # Initialize a matrix to store the edit distances
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]

    # Fill the first row and column with the number of insertions needed
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i

    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j

    # Iterate through the prediction and reference tokens
    for i, p_tok in enumerate(prediction_tokens):
        for j, r_tok in enumerate(reference_tokens):
            # If the tokens are the same, the edit distance is the same as the previous entry
            if p_tok == r_tok:
                dp[i + 1][j + 1] = dp[i][j]
            # If the tokens are different, the edit distance is the minimum of the previous entries plus 1
            else:
                dp[i + 1][j + 1] = min(dp[i][j + 1], dp[i + 1][j], dp[i][j]) + 1

    # Return the final entry in the matrix as the edit distance
    return dp[-1][-1]


def get_cer(
    preds: typing.Union[str, typing.List[str]],
    target: typing.Union[str, typing.List[str]],
) -> float:
    """Update the cer score with the current set of references and predictions.

    Args:
        preds (typing.Union[str, typing.List[str]]): list of predicted sentences
        target (typing.Union[str, typing.List[str]]): list of target words

    Returns:
        Character error rate score
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer


def get_wer(
    preds: typing.Union[str, typing.List[str]],
    target: typing.Union[str, typing.List[str]],
) -> float:
    """Update the wer score with the current set of references and predictions.

    Args:
        target (typing.Union[str, typing.List[str]]): string of target sentence or list of target words
        preds (typing.Union[str, typing.List[str]]): string of predicted sentence or list of predicted words

    Returns:
        Word error rate score
    """
    if isinstance(preds, str) and isinstance(target, str):
        preds = [preds]
        target = [target]

    if isinstance(preds, list) and isinstance(target, list):
        errors, total_words = 0, 0
        for _pred, _target in zip(preds, target):
            if isinstance(_pred, str) and isinstance(_target, str):
                errors += edit_distance(_pred.split(), _target.split())
                total_words += len(_target.split())
            else:
                print(
                    "Error: preds and target must be either both strings or both lists of strings."
                )
                return np.inf

    else:
        print(
            "Error: preds and target must be either both strings or both lists of strings."
        )
        return np.inf

    wer = errors / total_words

    return wer


def cer_beetween_files(f1_pred, f2_target):
    df_pred = pd.read_excel(f1_pred, names=["X", "Y", "O"], header=None)
    df_targ = pd.read_excel(f2_target, names=["X", "Y", "O"], header=None)
    df_targ["O"] = df_targ["O"].apply(
        lambda x: str(x).replace("k", "К") if not str(x) == "nan" else x
    )
    df_targ["O"] = df_targ["O"].apply(
        lambda x: str(x).replace("к", "К") if not str(x) == "nan" else x
    )
    df_targ["O"] = df_targ["O"].apply(
        lambda x: str(x).replace("д", "К") if not str(x) == "nan" else x
    )
    df_targ["O"] = df_targ["O"].apply(
        lambda x: str(x).replace("Д", "К") if not str(x) == "nan" else x
    )
    print(len(df_pred), len(df_targ))
    # indexs = np.where(df_pred['O'] == "К")[0]
    start = 0
    X_list_target = []
    Y_list_target = []
    for end in np.where(df_targ["O"] == "К")[0].tolist() + [len(df_targ)]:
        # if (df_targ.iloc[start,0] == df_targ.iloc[end - 1,0]).all():
        #     X_list = [str(x) for x in df_targ.iloc[start:end-1].X]
        #     Y_list = [str(x) for x in df_targ.iloc[start:end-1].Y]
        # else:
        #     X_list = [str(x) for x in df_targ.iloc[start:end].X]
        #     Y_list = [str(x) for x in df_targ.iloc[start:end].Y]
        if df_targ.iloc[start, 0] == df_targ.iloc[end - 1, 0]:
            X_list = [str(x) for x in df_targ.iloc[start : end - 1].X]
            Y_list = [str(x) for x in df_targ.iloc[start : end - 1].Y]
        else:
            X_list = [str(x) for x in df_targ.iloc[start:end].X]
            Y_list = [str(x) for x in df_targ.iloc[start:end].Y]

        start = end
        X_list_target = X_list_target + X_list
        Y_list_target = Y_list_target + Y_list

    indexs = np.where(df_pred["O"] == "К")[0]
    start = 0
    X_list_pred = []
    Y_list_pred = []
    for end in indexs.tolist() + [len(df_pred)]:
        if end == len(df_pred):
            X_list = [str(x) for x in df_pred.iloc[start : end - 1].X]
            Y_list = [str(x) for x in df_pred.iloc[start : end - 1].Y]
        else:
            X_list = [str(x) for x in df_pred.iloc[start : end - 1].X]
            Y_list = [str(x) for x in df_pred.iloc[start : end - 1].Y]

        start = end
        X_list_pred = X_list_pred + X_list
        Y_list_pred = Y_list_pred + Y_list

    cer_x = get_cer(X_list_pred, X_list_target)
    cer_y = get_cer(Y_list_pred, Y_list_target)
    cer = (cer_x + cer_y) / 2
    return (
        cer,
        cer_x,
        cer_y,
        len(X_list_pred),
        len(X_list_target),
        X_list_pred,
        X_list_target,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target_path",
        default="../dataset/tabl/",
        type=str,
        help="path to target data folder",
    )
    parser.add_argument(
        "--predict_path",
        default="./results/",
        type=str,
        help="path to predict data folder",
    )
    args = parser.parse_args()
    print(args.target_path)
    files_target = {x.stem: x for x in list(Path(args.target_path).rglob("*.xlsx"))}

    files = {x.stem: x for x in list(Path(args.predict_path).glob("*.xlsx"))}
    files_predict = {}
    for k, v in files.items():
        if "Плохое" in k:
            name = k.split("_")[1]
            files_predict[k.split("_")[1]] = v
        else:
            files_predict[k] = v

    print(len(files_target), len(files_predict))
    error = []

    results = {}
    for file in files_target:
        if not file in files_predict:
            continue
        print(files_target[file])
        try:
            cer, cer_x, cer_y, len_p, len_t, _, _ = cer_beetween_files(
                files_predict[file], files_target[file]
            )
        except Exception as e:
            print(e)
            continue
        results[file] = {
            "cer": "{:.2f}".format(cer),
            "len_X_pred": len_p,
            "len_X_target": len_t,
        }
        error.append(cer)

    df = pd.DataFrame(results).T
    df.to_csv("cer_table_fixf2.csv")
