import json
import functools
import multiprocessing
import time
import random
import os

import numpy as np

from app.tfrecord import SaveBuffer
from app.pipeline import PIPELINES
from app.assist import to_target_integers, split_indices
from app.glob import *

# ------------------------------------------------------------------------------------------------
#
#                                       HELPER FUNCTIONS
#
# ------------------------------------------------------------------------------------------------

def single_label_save(filename, data, labels, identifier, meta, buffers, target_list):
    label_dict = to_target_integers(target_list, labels)

    meta_dict = {
        TF_DATASET_NAME: filename.split("/")[-3],
        TF_FILENAME: filename,
    }

    buffers[identifier].add(data.astype(np.float32), {**label_dict, **meta_dict})


# ------------------------------------------------------------------------------------------------
#
#                                       MAIN PROCESSES
#
# ------------------------------------------------------------------------------------------------

def load_filenames(in_dirs, proc_name, subset=None):
    for in_dir in in_dirs:
        if not os.path.isdir(in_dir):
            raise ValueError(f"input directory {in_dir} does not exist")

    all_files = []

    for _, in_dir in enumerate(in_dirs):
        dir_files = [in_dir +  f for f in os.listdir(in_dir) if f in subset]
        all_files.extend(dir_files)

    print(f"total number of training samples: {len(all_files)}")

    return all_files

def process_trfrecords(workers, out_parent_dir, proc_name, records_per_file, 
    target_list, dimensions, in_dirs, pipeline, subset, **kwargs):
    save_dir = out_parent_dir + proc_name + "/"

    buffers = [SaveBuffer(save_dir, records_per_file, i) for i in range(workers)]
    queue = multiprocessing.Queue(workers)

    out_dir = out_parent_dir + proc_name + "/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    all_files = load_filenames(in_dirs, proc_name, subset)

    procs = process_data(
        filenames=all_files,
        pipeline=PIPELINES[pipeline],
        save_fn=lambda filename, data, labels, identifier, meta: single_label_save(filename, data, labels, identifier, meta, buffers=buffers, target_list=target_list),
        end_fn=lambda identifier: queue.put(buffers[identifier]),
        workers=workers,
        **kwargs
    )

    while not queue.full():
        time.sleep(0.5)
    time.sleep(0.5)

    final_buffer = SaveBuffer(save_dir, records_per_file, workers)
    while not queue.empty():
        b = queue.get()
        for img, label in b.empty_out():
            final_buffer.add(img, label)

    print(f"{len(final_buffer)} leftover groups (records_per_file being {records_per_file})")

    metadata = {
        "records_per_file": records_per_file,
        "dimensions": dimensions,
        "target_list": target_list,
        "final_file_name": final_buffer.next_filename(),
        "final_file_count": len(final_buffer)
    }

    final_buffer.flush()

    for p in procs:
        p.terminate()

    with open(save_dir + "meta.json", 'w') as fp:
        json.dump(metadata, fp)

def process_data(filenames, load_fn, pipeline, save_fn, end_fn, workers=1, assertions=[]):
    print("Beginning Data Processing\n")

    def process(index_subset, identifier):
        for i, filename in enumerate(index_subset):
            print("processing {} ({})".format(i, filename))
            
            print(filename)
            sample, labels = load_fn(filename)
            sample, labels, meta = pipeline.apply(sample, labels)

            save_fn(filename, sample, labels, identifier, meta)
        
        time.sleep(0.5)
        end_fn(identifier)
        time.sleep(0.5)
        
    if workers == 1:
        process(filenames, 0)
        return []
    else:
        print("number of files to process: {}".format(len(filenames)))
        index_split = split_indices(filenames, workers)

        procs = []

        for i, subset in enumerate(index_split):
            p = multiprocessing.Process(target=process, args=(subset, i),  daemon=True)
            p.start()
            procs.append(p)

        return procs
    print("Data Processing Complete\n")