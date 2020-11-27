import random
import math
import functools

import cv2
from matplotlib import cm
from PIL import Image, ImageEnhance
import numpy as np

from app.assist import *

class Pipeline():
    def __init__(self, processes, seed=987):
        self.processes = processes
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def apply(self, sample, labels):
        meta = {} # for saving data generated inside procs at runtime, in case this is needed.

        for proc in self.processes:
            sample, labels = proc(sample, labels, meta)

        return sample, labels, meta

class NoopPipeline(Pipeline):
    def __init__(self):
        super().__init__([])
    
    def apply(self, sample, labels):
        return sample, labels, {}

# ------------------------------------------------------------------------------------------------------------------
#
#                                                   IMAGE PROCS
#
# ------------------------------------------------------------------------------------------------------------------

def sharpen(s, label, meta):
    s = normalize_relative(np.squeeze(s))
    s = Image.fromarray(np.uint8(s*255))
    e = ImageEnhance.Sharpness(s.convert("RGB"))
    s = e.enhance(4)

    return normalize_relative(np.array(s)), label

def rand_crop_between(s, label, min_width, max_width, min_height, max_height):
    return rand_crop(s, label, (np.random.randint(min_height, max_height), np.random.randint(min_width, max_width)))

def rand_crop(s, label, dimensions):
    x_offset = np.random.randint(0, s.shape[1]-dimensions[1])
    y_offset = np.random.randint(0, s.shape[0]-dimensions[0])
    return s[y_offset:y_offset+dimensions[0], x_offset:x_offset+dimensions[1], :], label

def crop(s, labels, xstart, xend, ystart, yend):
    return s[ystart:yend, xstart:xend, :], labels

def noop(s, lab, meta):
    return s, lab

def make_random_crop_proc(dimensions):
    return lambda s, label, meta: rand_crop(s, label, dimensions)

def normalize_rgb(s, labels, meta):
    return s * 255.0/s.max(), labels  

def scale_one_neg_one(img):
    img_min = img.min()
    
    return (2 * ((img - img_min) / (img.max() - img_min))) - 1

def normalize_one_neg_one(s, labels, meta):
    return scale_one_neg_one(s), labels

def resize(s, label, meta, x, y):
    s = cv2.resize(s, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

    return s, label  

def grayscale(s, labels, meta):
    return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), labels  

def single_channel(s, label, meta):
    return s[:, :, 0], label

def random_flip(s, labels, meta, x, y):
    s = cv2.resize(s, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

    return s, labels  

PIPELINES = {
    "256x256": Pipeline([
        lambda s, labels, meta: resize(s, labels, meta, 256, 256),
    ])
}
