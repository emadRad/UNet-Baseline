import os
import torch
import torchvision
from torch.utils import data
import imageio
import cv2 as cv
import numpy as np

# TIFFSetField: tempfile.tif: Unknown pseudo-tag 65538. for reading .gif labels
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor, Normalize


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def get_anno_path(img_name, anno_path):
    img_name = img_name.split("_")[0]+'_manual1.gif'
    return os.path.join(anno_path, img_name)


def get_drive_filename_pairs(image_path, anno_path):
    imgs_full_path = recursive_glob(image_path, suffix=".tif")
    pairs = []
    for img_full_path in imgs_full_path:
        anno_full_path = get_anno_path(os.path.basename(img_full_path), anno_path)
        pairs.append((img_full_path, anno_full_path))

    return pairs


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class DRIVE(data.Dataset):

    DRIVE_ROOT_FOLDER_NAME = "DRIVE"

    def __init__(
            self,
            root=None,
            train=True,
            img_transform=None,
            lbl_transform=None):

        self.root = root
        self.drive_full_path = os.path.join(self.root, self.DRIVE_ROOT_FOLDER_NAME)

        self.img_transform = img_transform
        self.lbl_transform = lbl_transform

        if train:
            self.drive_images_path = os.path.join(self.drive_full_path, "training", "images")
            self.drive_anno_path = os.path.join(self.drive_full_path, "training", "1st_manual")
            self.img_anno_pairs = get_drive_filename_pairs(self.drive_images_path, self.drive_anno_path)

        # print(self.img_anno_pairs[0])
        # img = cv.imread(self.img_anno_pairs[0][0])
        # lbl = imageio.imread(self.img_anno_pairs[0][1])
        # print(img.shape)


    def __len__(self):
        return len(self.img_anno_pairs)


    def __getitem__(self, index):
        img_path = self.img_anno_pairs[index][0]
        anno_path = self.img_anno_pairs[index][1]

        img = cv.imread(img_path)
        anno = imageio.imread(anno_path)
        anno = np.asarray(anno)

        anno = anno / 255

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.lbl_transform is not None:
            anno = self.lbl_transform(anno)

        return img, anno


if __name__ == "__main__":

    trans = torchvision.transforms.Compose([
        ToTensor(),
        Normalize([.485, .456, .406],[.229, .224, .225])
    ])

    lbl_transform = torchvision.transforms.Compose([
        ToLabel()
    ])

    drive = DRIVE(root="../datasets/", img_transform=trans, lbl_transform=lbl_transform)

    img, lbl = next(iter(drive))

    # print(img.shape)
    print(lbl.shape)
    
    