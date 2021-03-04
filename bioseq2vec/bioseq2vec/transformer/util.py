'''utility functions for data transformation'''
import numpy as np
from sklearn.preprocessing import normalize

from yoctol_utils.hash import consistent_hash

def one_hot_encode_seq(seq, max_index):
    encoded_seq = []
    for index in seq:
        arr = np.zeros(max_index)
        arr[index] = 1
        encoded_seq.append(arr)
    return encoded_seq


def hash_seq(sequence, max_index):
    return [consistent_hash(word) % max_index + 1 for word in sequence]
