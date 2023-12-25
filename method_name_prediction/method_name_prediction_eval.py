from functools import lru_cache
from typing import List
import numpy as np


def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    """
    if not len(camel_case_identifier):
        return []

    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    snake_case = identifier.split("_")

    identifier_parts = []
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(s.lower() for s in split_camelcase(part))
    if len(identifier_parts) == 0:
        return [identifier]
    identifier_parts = [x for x in identifier_parts if x]
    return identifier_parts

@lru_cache(maxsize=5000)
def split_identifier_into_parts_with_idx(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    snake_case = identifier.split("_")

    identifier_parts = []
    idx = 0
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            for s in split_camelcase(part):
                # print(i, part, s)
                identifier_parts.append((s.lower(), idx, idx + len(s)))
                idx += len(s)
    if len(identifier_parts) == 0:
        return [identifier]
    identifier_parts = [x for x in identifier_parts if x]
    return identifier_parts


def calculate_precision(prediction, ground_truth):
    ground_truth_splits = split_identifier_into_parts(ground_truth)
    prediction_splits = split_identifier_into_parts(prediction)
    correct_results = 0
    all_returned_results = 0
    for split in prediction_splits:
        if split in ground_truth_splits:
            correct_results += 1
        all_returned_results += 1
    precision = float(correct_results/all_returned_results)
    return precision


def calculate_recall(prediction, ground_truth):
    ground_truth_splits = split_identifier_into_parts(ground_truth)
    should_have_returned_results = len(ground_truth_splits)
    prediction_splits = split_identifier_into_parts(prediction)
    correct_results = 0
    for split in prediction_splits:
        if split in ground_truth_splits:
            correct_results += 1

    recall = float(correct_results/should_have_returned_results)
    return recall


def calculate_f1_score(prediction, ground_truth):
    precision = calculate_precision(prediction, ground_truth)
    recall = calculate_recall(prediction, ground_truth)
    f1_score = 0
    if precision > 0 and recall > 0:
        f1_score = float(2*(precision * recall)/(precision + recall))
    return f1_score, precision, recall


def calculate_recalls(predictions, ground_truths):
    recalls = []
    for i, prediction in enumerate(predictions):
        recall = calculate_recall(prediction, ground_truths[i])
        recalls.append(recall)

    return np.mean(recalls)


def calculate_precisions(predictions, ground_truths):
    precisions = []
    for i, prediction in enumerate(predictions):
        precision = calculate_precision(prediction, ground_truths[i])
        precisions.append(precision)

    return np.mean(precisions)


def calculate_f1_scores(predictions, ground_truths):
    f1_scores = []
    precisions, recalls = [], []
    for i, prediction in enumerate(predictions):
        f1_score, precision, recall = calculate_f1_score(prediction, ground_truths[i])
        f1_scores.append(f1_score)
        recalls.append(recall)
        precisions.append(precision)

    return np.mean(f1_scores), np.mean(precisions), np.mean(recalls)


predictions = ['putString1', 'computeTriangles', 'computeTriangles', 'computeTriangles',
               'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles',
               'computeTriangles', 'computeTriangles', 'computeTriangles', 'computeTriangles']
ground_truths = ['putString', 'remove', 'getSoundSource', 'loop', 'pause',
                 'setLooping', 'add', 'setVolume', 'dispose', 'clear', 'setPitch', 'main']
