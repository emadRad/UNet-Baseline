import random
from PIL import Image, ImageOps
import cv2 as cv
import torchvision.transforms.functional as tf
import numpy as np




class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask, weight):
        assert isinstance(img, np.ndarray), "img should be a numpy ndarray."

        assert img.shape[:2] == mask.shape[:2]
        for a in self.augmentations:
            img, mask, weight = a(img, mask, weight)

        return img, mask, weight




class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, weight):
        assert img.size == mask.size, "Image and labels are not in the same size."
        w, h = img.size

        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
                weight.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
                weight.resize((ow, oh), Image.NEAREST),

            )



class RandomHorizontallyFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, mask, weight):
        if random.random() < self.prob:
            return (
                cv.flip(img,  flipCode=1),
                cv.flip(mask, flipCode=1),
                cv.flip(weight, flipCode=1)
            )
        return img, mask, weight


# rotate with a random degree from (-degree, degree)
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, weight):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        h, w = img.shape[:2]
        image_center = (h//2, w//2)
        rotation_mat = cv.getRotationMatrix2D(image_center, rotate_degree, 1)
        return (
            cv.warpAffine(img, rotation_mat, (w,h)),
            cv.warpAffine(mask, rotation_mat, (w,h)),
            cv.warpAffine(weight, rotation_mat, (w, h))
        )


augmentation_dict = {
        "hflip": RandomHorizontallyFlip,
        'rotate': RandomRotate,
}


def get_augmentations(augs):

    augmentations = []
    print("Augmentations:")
    for aug, param in augs.items():
        augmentations.append(augmentation_dict[aug](param))
        print("{}: {}".format(aug, param))

    return Compose(augmentations)




