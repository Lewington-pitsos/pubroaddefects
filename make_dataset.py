import argparse
import json
import os


from app.process import *
from app.glob import *
from app.assist import timed, load_image, load_subset
from app.tfrecord import *

class TrickyValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", help="number of workers to run at once", type=int,default=1)

args = parser.parse_args()

samples = {}

for name, data in DATASETS.items():
    samples[name] = os.listdir(data["path"])

SUBSETS = {
    "all": samples["japan"] + samples["india"] + samples["czech"] + samples["usa"] + samples["italymexico"] + samples["indonesia"],
    "big_5": load_subset("big_5"), 
}

param_queue = []
with open(PROCESS_QUEUE + "next.json") as f:
    param_queue += json.load(f)["queue"]

for params in param_queue:
    print(params)
    
    records_per_file = params["records_per_file"] if "records_per_file" in params else 80

    target_list = TARGET_LISTS[params["target_list"]]

    print("number of classes in dataset:", len(target_list))

    timed(
        process_trfrecords,

        # process_tfrecords args
        workers=args.workers,
        out_parent_dir=PROCESSED_DATA_PATH,
        proc_name=params["name"],
        records_per_file=records_per_file,
        target_list=target_list,
        dimensions=params["dimensions"],

        # file processing args
        in_dirs=[a["path"] for a in DATASETS.values()],
        load_fn=load_image,
        pipeline=params["pipeline"],
        subset=SUBSETS[params["subset"]],
    )

    with open(PROCESS_QUEUE + "saved.json") as f:
            saved = json.load(f)  

    saved["saved"].append(params)

    with open(PROCESS_QUEUE + "saved.json", 'w') as fp:
        json.dump(saved, fp, indent=4, cls=TrickyValuesEncoder)
