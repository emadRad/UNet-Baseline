import os
import torch


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def one_hot_encoding(labels):

    classes_num = len(labels.unique())
    h , w = labels.shape[-2:]

    mask = torch.zeros(classes_num, h, w)

    for c in range(classes_num):
        mask[c, :, :] = (labels == c)

    return mask
