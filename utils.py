import os
import itertools

import torch
from torchvision import utils

from skimage import color
import numpy as np
import matplotlib.pyplot as plt

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


#TODO change it to the faster version like in the DiceLoss
def one_hot_encoding(labels):

    classes_num = len(labels.unique())
    h , w = labels.shape[-2:]

    mask = torch.zeros(classes_num, h, w)

    for c in range(classes_num):
        mask[c, :, :] = (labels == c)

    return mask


def plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name):
    f = plt.figure(figsize=(20, 20))
    N, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(131)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Slices')

    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(132)
    plt.imshow(color_grid)
    plt.title('Ground Truth')

    plt.subplot(133)
    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.imshow(color_grid)
    plt.title('Prediction')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight')
    plt.gcf().clear()



def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          file_save_name="temp.pdf"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)
    f = plt.figure(figsize=(35, 35))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    f.savefig(file_save_name, bbox_inches='tight')


    plt.gcf().clear()
