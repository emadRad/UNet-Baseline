import os

import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms

import numpy as np
import imageio
import cv2 as cv

from utils import one_hot_encoding, recursive_glob


def create_weight_mask(labels, maxWeight=5, maxEdgeWeight=5):
    '''
    Function to create weighted mask - with median frequency balancing and edge-weighting
    '''

    unique, counts = np.unique(labels, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > maxWeight] = maxWeight

    (h, w) = labels.shape

    weights_mask = np.reshape(class_wise_weights[labels.ravel()], (h, w))

    # Gradient Weighting
    (gx, gy) = np.gradient(labels)
    gradWeight = maxEdgeWeight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2), 0.5) > 0,
                                            dtype='float')

    weights_mask += gradWeight


    return weights_mask



class DRIVE(data.Dataset):

    DRIVE_ROOT_FOLDER_NAME = "DRIVE"

    def __init__(
            self,
            root=None,
            train=True,
            img_size=584,
            augmentations=None,
            one_hot_encoding=False
    ):

        self.root = root
        self.drive_full_path = os.path.join(self.root, self.DRIVE_ROOT_FOLDER_NAME)

        self.augmentations = augmentations

        if train:
            self.drive_images_path = os.path.join(self.drive_full_path, "training", "images")
            self.drive_lbl_path = os.path.join(self.drive_full_path, "training", "1st_manual")
            self.drive_mask_path = os.path.join(self.drive_full_path, "training", "mask")
            self.img_lbl_mask_triples = get_drive_filename_triples(self.drive_images_path,
                                                                   self.drive_lbl_path,
                                                                   self.drive_mask_path,
                                                                   mode="training")

        # it is testing data
        else:
            self.drive_images_path = os.path.join(self.drive_full_path, "test", "images")
            self.drive_lbl_path = os.path.join(self.drive_full_path, "test", "1st_manual")
            self.drive_mask_path = os.path.join(self.drive_full_path, "test", "mask")
            self.img_lbl_mask_triples = get_drive_filename_triples(self.drive_images_path,
                                                                   self.drive_lbl_path,
                                                                   self.drive_mask_path,
                                                                   mode="test")



        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )

        self.one_hot_encoding = one_hot_encoding

        self.img_transform = transforms.Compose([transforms.ToTensor()
                                                 ])


    def __len__(self):
        return len(self.img_lbl_mask_triples)


    def __getitem__(self, index):
        img_path  = self.img_lbl_mask_triples[index][0]
        lbl_path  = self.img_lbl_mask_triples[index][1]
        mask_path = self.img_lbl_mask_triples[index][2]

        img = cv.imread(img_path)
        lbl = imageio.imread(lbl_path)
        mask = imageio.imread(mask_path)
        lbl = np.asarray(lbl)
        mask = np.asarray(mask)

        lbl = lbl / 255
        mask = mask / 255

        # 1:vessels, 2:background
        lbl[mask == 0] = 2

        weight_mask = create_weight_mask(lbl.astype(np.int))

        if self.augmentations is not None:
            img, lbl, weight_mask = self.augmentations(img, lbl, weight_mask )

        im, lbl, weight_mask = self.transform(img, lbl, weight_mask)


        if self.one_hot_encoding:
            lbl = one_hot_encoding(lbl)

        return im, lbl, weight_mask


    def transform(self, img, lbl, weight_mask):

        img_resized = cv.resize(img, self.img_size, interpolation=cv.INTER_LINEAR)
        lbl_resized = cv.resize(lbl, self.img_size, interpolation=cv.INTER_NEAREST)
        weight_resized = cv.resize(weight_mask, self.img_size, interpolation=cv.INTER_NEAREST)

        img = self.img_transform(img_resized)

        weight_mask = torch.from_numpy(weight_resized).float()
        lbl = torch.from_numpy(np.array(lbl_resized)).long()

        return img, lbl, weight_mask


def get_lbl_mask_path(img_name, lbl_path, mask_path, mode):
    img_name = img_name.split("_")[0]
    lbl_name = img_name + '_manual1.gif'
    mask_name = img_name + '_{}_mask.gif'.format(mode)
    return os.path.join(lbl_path, lbl_name), os.path.join(mask_path, mask_name)


def get_drive_filename_triples(image_path, lbl_path, mask_path, mode):
    imgs_full_path = recursive_glob(image_path, suffix=".tif")
    triples = []
    for img_full_path in imgs_full_path:
        lbl_full_path, mask_full_path = get_lbl_mask_path(os.path.basename(img_full_path), lbl_path, mask_path, mode)
        triples.append((img_full_path, lbl_full_path, mask_full_path))

    return triples
