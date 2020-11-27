import os
import datetime
import json

from matplotlib import image
import numpy as np
import h5py
import random
from xml.etree import ElementTree

from app.glob import *

# ------------------------------------------------------------------------
#
#                                   ASSIST
#
# ------------------------------------------------------------------------

def timed(op, **kwargs):
    start = datetime.datetime.now()

    op(**kwargs)

    print("Operation Duration: {}".format(datetime.datetime.now() - start))

def normalize_relative(img):
    return normalize(img, img.min(), img.max())

def normalize(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)

def standardize_relative(x):
    return standardize(x, np.mean(x), np.std(x))

def standardize(x, mean, sd):
    return (x - mean) / sd

def cleaned_filename(filename):
    if filename[-4:] == ".mp3":
        filename = filename[:-4]
    
    return filename

def to_target_integers(target_index, labels, weights=None):
    objects = labels.findall('object')    

    label_dict = {
        TF_CLASS_LABEL: np.zeros(len(target_index), dtype=np.int16),
        TF_COUNT_LABEL: np.zeros(len(target_index), dtype=np.int16),

        # currently never used
        TF_OBJECT_LABEL: np.zeros(len(target_index), dtype=np.int16),
        TF_XMINS_LABEL: np.zeros(len(target_index), dtype=np.int16),
        TF_XMAXS_LABEL: np.zeros(len(target_index), dtype=np.int16),
        TF_YMINS_LABEL: np.zeros(len(target_index), dtype=np.int16),
        TF_YMAXS_LABEL: np.zeros(len(target_index), dtype=np.int16),
    }

    for object_index, obj in enumerate(objects):
        label = obj.find("name").text

        if label in target_index:
            label_index = target_index.index(label)
            label_dict[TF_CLASS_LABEL][label_index] = 1
            label_dict[TF_COUNT_LABEL][label_index] += 1

    return label_dict

def full_contents(path):
    return [path + f for f in os.listdir(path)]

def annotation_from_image(filename):
    return filename.replace("images", "annotations/xmls").replace(".jpg", ".xml").replace(".png", ".xml").replace(".jpeg", ".xml")

def label_from_image(filename):
    identifier = filename.split("/")[-1]
    start = identifier.split("_")[0]
    number = identifier.split("_")[-1].split(".")[0].zfill(5)
    
    return filename.replace("images", "labels").replace(identifier, f"{start}_{number}.png")
    
def image_from_annotation(filename):
    return filename.replace("annotations/xmls", "images",).replace(".xml", ".jpg")

def load_image(filename):
    data = image.imread(filename)

    with open(annotation_from_image(filename)) as xml_file:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
    
    return data, root

def get_dimensions(root):
    return int(root.find('size').find("width").text), int(root.find('size').find("height").text)

def offset_box_bounds_randomly(obj, width, height, min_offset, max_offset):
    return offset_box_bounds(obj, width, height, random.randint(min_offset, max_offset), random.randint(min_offset, max_offset), random.randint(min_offset, max_offset), random.randint(min_offset, max_offset))

def offset_box_bounds(obj, width, height, xmin_offset, xmax_offset, ymin_offset, ymax_offset):
    xmin, xmax, ymin, ymax = extract_box_bounds(obj, width, height)
    xmin = max(0, xmin - xmin_offset)
    xmax = min(width, xmax + xmax_offset)
    ymin = max(0, ymin - ymin_offset)
    ymax = min(height, ymax + ymax_offset)

    return xmin, xmax, ymin, ymax

def extract_box_bounds(obj, width, height):
    xmlbox = obj.find('bndbox')
    xmin = max(0, int(xmlbox.find('xmin').text))
    xmax = min(width, int(xmlbox.find('xmax').text))
    ymin = max(0, int(xmlbox.find('ymin').text))
    ymax = min(height, int(xmlbox.find('ymax').text))

    return xmin, xmax, ymin, ymax


def split_indices(indices, num_splits):
    assert num_splits > 0

    if num_splits == 1:
        return [indices]

    index_segments = []
    count = len(indices)
    to_each_worker = round(count / num_splits - 1)

    for i in range(num_splits - 1):
        end_idx = i* to_each_worker + to_each_worker
        index_segments.append(indices[i * to_each_worker:end_idx])

    index_segments.append(indices[end_idx:])
    
    return index_segments

# ----------------------------------------------------------------------------------
#
#                                SUBSET MANAGEMENT
#
# ----------------------------------------------------------------------------------


def load_subset(name):
    with open(SUBSET_DIR + name + ".json") as f:
        return json.load(f)["subset"]

def create_subset(name, values):
    with open(f"subsets/{name}.json", 'w') as f:
        json.dump({"subset": values}, f)