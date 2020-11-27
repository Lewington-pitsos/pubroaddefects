import functools
import json
import math
import random

import tensorflow as tf
import numpy as np

from app.glob import *
from app.maths import *
from app.assist import full_contents

def add_count(img, label, max_count=0):  
    return img, (label, tf.one_hot(tf.math.reduce_sum(label, axis=1), max_count))

def make_smoothen(max_label):
    return lambda img, label: (img, tf.clip_by_value(tf.cast(label, tf.float32), 0, max_label))

def noop_process(img, label):
    return img, label

def standardize(img, label):
    return tf.image.per_image_standardization(img), label

def grayscale(img, label):
    return tf.image.rgb_to_grayscale(img), label

def make_resize(size):
    def resize(img, label):
        return tf.image.resize(img, size), label
    return resize

def binarize(img, label):
    return img, tf.one_hot(tf.cast(label[0], tf.uint8), 2, on_value=0, off_value=1)    

def chain(procs):
    def apply(img, label):
        for p in procs:
            img, label = p(img, label)

        return img, label

    return apply

def bot_adjust(img, label):
    out = tf.image.random_brightness(img, 0.4, seed=None)
    out = tf.image.random_hue(out, 0.4, seed=None)
    out = tf.image.random_saturation(out, 0.6, 1.4, seed=None)

    return out, label
    
def make_bot_normalize():
    means = tf.constant([[[123.68, 116.779, 103.939]]])
    stds = tf.constant([[[58.393, 57.12, 57.375]]])

    def bot_normalize(img, label):
        return (img - means) / stds

    return bot_normalize 

# Augmentations -----------------------------------------------------------------------------------

def normalize_process(img, label):
    return tf.math.l2_normalize(img), label

def flip_img(img, label):    
    return tf.image.random_flip_left_right(img, seed=None), label

def random_brightness(img, label):
    return tf.image.random_brightness(img, 0.9, seed=None), label

def random_contrast(img, label):
    return tf.image.random_contrast(img, 0.5, 1.0, seed=None), label

def random_saturation(img, label):
    return tf.image.random_saturation(img, 0.4, 1.0, seed=None), label

def random_hue(img, label):
    return tf.image.random_hue(img, 0.4, seed=None), label

def make_random_crop(size):
    def random_crop(img, label):
        return tf.image.random_crop(img, size, seed=None), label
    return random_crop

def make_center_crop(proportion):
    def center_crop(img, label):
        return tf.image.central_crop(img, proportion), label

    return center_crop

def make_gaussian_noise(stddev):
    noiser = tf.keras.layers.GaussianNoise(stddev)

    def apply_noise(img, label):
        return noiser(img), label
    
    return apply_noise

def random_jpeg_quality(img, label):
    return tf.image.random_jpeg_quality(img, 40, 100, seed=None), label


static_load_procs = {
    "normalize": normalize_process,
    "bot_normalize": make_bot_normalize(),
    "add_count": functools.partial(add_count, max_count=MAX_DEFECTS_PER_IMAGE),
    "binarize": binarize,
    "smooth95": make_smoothen(0.95),
    "smooth98": make_smoothen(0.98),
    "standardize_img": standardize,
    "std_grayscale": chain([standardize, grayscale]),
    "grayscale": grayscale
}

static_aug_procs = {
    "bot_adjust": (chain([make_center_crop(0.875), bot_adjust]), make_center_crop(0.875)),

    "random_flip": (flip_img, None),
    "random_crop_flip320": (chain([flip_img, make_random_crop((320, 320, 3))]),  make_resize((320, 320))),
    "random_crop_flip224": (chain([flip_img, make_random_crop((224, 224, 3))]),  make_resize((224, 224))),
    "random_crop_flip_gauss224": (chain([flip_img, make_random_crop((224, 224, 3)), make_gaussian_noise(0.3)]),  make_resize((224, 224))),
    "random_crop_flip420": (chain([flip_img, make_random_crop((420, 420, 3))]),  make_resize((420, 420))),
    "random_crop_flip400": (chain([flip_img, make_random_crop((400, 400, 3))]),  make_resize((400, 400))),
    "random_crop_flip380": (chain([flip_img, make_random_crop((380, 380, 3))]),  make_resize((380, 380))),
    "random_crop_flip530": (chain([make_resize((600, 600)), flip_img, make_random_crop((530, 530, 3))]),  make_resize((530, 530))),

    "random_flip_brightness": (chain([flip_img, random_brightness]), None),
    "random_flip_brightness_contrast": (chain([flip_img, random_brightness, random_contrast]), None),
    "random_safe_aug": (chain([
        flip_img, 
        random_brightness, 
        random_contrast,
        random_hue,
        random_jpeg_quality,
        random_saturation,
    ]), None),
    "random_flip_saturation": (chain([flip_img, random_saturation]), None),
    "random_flip_contrast": (chain([flip_img, random_saturation]), None),
    "random_brightness_contrast": (chain([random_brightness, random_contrast]), None),
    "random_saturation_contrast": (chain([random_saturation, random_contrast]), None),
    "random_saturation": (random_saturation, None),
    "random_contrast": (random_contrast, None),
    "random_brightness": (random_brightness, None),
    "random_hue": (random_hue, None),
    "random_jpg": (random_jpeg_quality, None),
    "random_crop_320": (make_random_crop((320, 320, 3)), make_resize((320, 320))),
    "random_crop_420": (make_random_crop((420, 420, 3)), make_resize((420, 420))),
}

def set_label_shape(label, output_range):
    if isinstance(output_range, list) and isinstance(label, tuple):
        for i, l in enumerate(label):
            l.set_shape(output_range[i])
    elif isinstance(label, tuple):
        for l in label:
            l.set_shape(output_range)            
    else:
        label.set_shape(output_range) 

def make_set_shape_fn(input_dims, output_range):
    def fixup_shape(img, label_data):
        img.set_shape(input_dims)
        
        if isinstance(label_data, dict):
            set_label_shape(label_data[TF_CLASS_LABEL], output_range)
            set_label_shape(label_data[TF_COUNT_LABEL], output_range)
        else:
            set_label_shape(label_data, output_range)

        return img, label_data
    
    return fixup_shape

def create_dataset(filenames, preprocess, aug, fixup_shape_fn, parse_fn, autotune=True):
    if preprocess is None:
        preprocess = noop_process
    if aug is None:
        aug = noop_process

    num_parallel = tf.data.experimental.AUTOTUNE if autotune else 1

    proc = tf.data.TFRecordDataset(filenames)\
        .map(parse_fn, num_parallel_calls=num_parallel, deterministic=False)\
        .map(preprocess, num_parallel_calls=num_parallel, deterministic=False)\
        .map(fixup_shape_fn, num_parallel_calls=num_parallel, deterministic=False)\
        .map(aug, num_parallel_calls=num_parallel, deterministic=False)

    return proc

def prepend_mixup(aug, mixup_dataset, alpha):
    itr = iter(mixup_dataset)    
    
    @tf.function
    def pure_mixup(img, label):
        mixup_img, mixup_label = next(itr)
        weight = np.random.beta(alpha, alpha)
        remainder = 1 - weight

        return img * weight + mixup_img * remainder, tf.cast(label, tf.float32) * weight + tf.cast(mixup_label, tf.float32) * remainder
    
    @tf.function
    def mixup_aug(img, label):
        img, label = pure_mixup(img, label)

        return aug(img, label)

    if aug is not None:
        return mixup_aug

    return pure_mixup

def transformed_dataset(filenames, batch_size, epochs, fixup_shape_fn, records_per_file, 
    final_file_count, final_file_name, parse_fn, mixup_filenames=None, preprocess=None, 
    aug=None, autotune=True, mixup_alpha=0.2):

    if final_file_count == 0:
        final_file_count = records_per_file

    if mixup_filenames is not None:
        mixup_data = create_dataset(mixup_filenames, preprocess, aug=None, fixup_shape_fn=fixup_shape_fn, parse_fn=parse_fn, autotune=autotune)\
            .shuffle(150, reshuffle_each_iteration=True)\
            .repeat()
        
        aug = prepend_mixup(aug, mixup_data, mixup_alpha)

    proc = create_dataset(filenames, preprocess, aug, fixup_shape_fn, parse_fn, autotune)

    prefetch_amt = tf.data.experimental.AUTOTUNE if autotune else 1

    steps = (len(filenames) - 1) * records_per_file

    for filename in filenames:
        if final_file_name in filename:
            steps += final_file_count
            break
    else:
        steps += records_per_file

    steps = int(np.floor(steps / batch_size))

    return {
        "data": proc.batch(batch_size).repeat(epochs).prefetch(prefetch_amt),
        "steps": steps
    }
