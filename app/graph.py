from matplotlib.pyplot import hexbin
import cv2
import random

import matplotlib.pyplot as plt

from app.assist import annotation_from_image, get_dimensions, offset_box_bounds_randomly
from xml.etree import ElementTree

def with_boxes(filename, mapping=None):
    img = cv2.imread(filename)
    annotation_name = annotation_from_image(filename)
    
    with open(annotation_name) as xml_file:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        img = add_boxes_to(img, root, mapping)

    return img[...,::-1]

def add_boxes_to(img, root, mapping=None):
    width, height = get_dimensions(root)

    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        
        if mapping is not None:
            cls_name = mapping[cls_name]

        xmin, xmax, ymin, ymax = offset_box_bounds_randomly(obj, width, height, 0, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # put text
        cv2.putText(img,cls_name,(xmin,ymin-10),font,1,(0,255,0),1,cv2.LINE_AA)

        # draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0),1)

    return img
