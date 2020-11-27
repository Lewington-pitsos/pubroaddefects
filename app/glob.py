DATA_PATH = "data/"
PROCESSED_DATA_PATH = "data/processed/"
SEED = 7896
PROCESS_QUEUE = "to_process/"
SUBSET_DIR = "subsets/"

SAVE_EVERY_EPOCH = "epoch"
SAVE_LAST = "last"
SAVE_BEST = "best"
SAVE_BEST_AND_LAST = "best_last"

COMMON_DEFECTS_MAP = {
    "D00": "Lateral Crack",
    "D10": "Longitudinal Crack",
    "D20": "Crockodile Crack",
    "D40": "Rutting, Bump, Pothole",
    "D43": "Blurred Zebra Crossing", 
    "D44": "Blurred White Line", 
}


TARGET_LISTS = {
    "all_defects": ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D50', 'Sealed_crack'],
    "big_5": ['D00', 'D10', 'D20', 'D40', 'D44'],
}

TF_CLASS_LABEL = 'label/classes'
TF_COUNT_LABEL = 'label/counts'
TF_OBJECT_LABEL = 'label/object_classes'
TF_XMINS_LABEL = 'label/xmins'
TF_XMAXS_LABEL = 'label/xmaxs'
TF_YMINS_LABEL = 'label/ymins'
TF_YMAXS_LABEL = 'label/ymaxs'

TF_DATASET_NAME = 'meta/dataset'
TF_FILENAME = 'meta/filename'

TF_IMAGE = "image"

TF_TENSORS = [
    TF_CLASS_LABEL,
    TF_COUNT_LABEL,
    TF_OBJECT_LABEL,
    TF_XMINS_LABEL,
    TF_XMAXS_LABEL,
    TF_YMINS_LABEL,
    TF_YMAXS_LABEL,
]

TF_STRINGS = [
    TF_DATASET_NAME,
    TF_FILENAME,
]

MAX_DEFECTS_PER_IMAGE = 15

# Data Management

DATASETS = {
    "japan": {
        "path": DATA_PATH+"train/Japan/images/",
    },
    "india": {
        "path": DATA_PATH+"train/India/images/",
    },
    "czech": {
        "path": DATA_PATH+"train/Czech/images/"
    },
    "usa": {
        "path": DATA_PATH+"train/USA/images/",
    },    
    "italymexico": {
        "path": DATA_PATH+"train/ItalyMexico/images/",
    },
    "indonesia": {
        "path": DATA_PATH+"train/Indonesia/images/",
    }
}

TRAINING_DATASET_NAMES = [
    "japan",
    "india",
    "czech",
    "usa",
    "italymexico",
    "indonesia"
]
TRAINING_DATASETS = {}

for name in TRAINING_DATASET_NAMES:
    TRAINING_DATASETS[name] = DATASETS[name] 