
import numpy as np
from collections import Counter

# lab.py
import torch

import numpy as np

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate entropy of the dataset.
    Assumes the last column is the target class.
    """
    target = data[:, -1]  # last column = class labels
    values, counts = np.unique(target, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-9))  # add epsilon to avoid log(0)
    return entropy

#PES2UG23CS348  MOHAMMED SHAZI

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the expected entropy (average information) 
    when splitting the dataset on a given attribute.
    """
    attribute_values = data[:, attribute]
    values, counts = np.unique(attribute_values, return_counts=True)
    total = len(data)
    avg_entropy = 0.0

    for v, count in zip(values, counts):
        subset = data[attribute_values == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_entropy += (count / total) * subset_entropy

    return avg_entropy



def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Information Gain = Dataset Entropy - Average Info of Attribute
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    return dataset_entropy - avg_info


def get_selected_attribute(data: np.ndarray):
    """
    Returns:
      - information_gains: dict {attribute: gain}
      - selected_attribute: attribute index with max gain
    """
    n_features = data.shape[1] - 1  # exclude target column
    info_gains = {}

    for attr in range(n_features):
        gain = get_information_gain(data, attr)
        info_gains[attr] = gain

    selected_attribute = max(info_gains, key=info_gains.get)
    return info_gains, selected_attribute
