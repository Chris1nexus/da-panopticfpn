#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image
import argparse
from panopticapi.utils import save_json
from label_reduced import labels, id2label, trainId2label


IGNORE_VALUE = 255
def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + IGNORE_VALUE
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, args):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    categories_list = []
    # define which set of ids to use for the panoptic task
    if args.use_train_id:
        id2label_dict = trainId2label
    else:
        id2label_dict = id2label
    # parse id set from cityscapes like to the one used by coco environments
    for id, label_info in id2label_dict.items(): 
        if label_info.ignoreInEval:
                continue
        categories_list.append({'id': int(label_info.trainId) if args.use_train_id else int(label_info.id),
                               'name': label_info.name,
                               'color': label_info.color,
                               'supercategory': label_info.category,
                               'isthing': 1 if label_info.hasInstances else 0})
    categories = {category['id']: category for category in categories_list}
    categories_list = sorted(categories_list, key=lambda x:x['id'])


    stuff_ids = [k["id"] for k in categories_list if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories_list if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0


    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(args.nproc , 1))
    print(f'Starting conversion with {args.nproc} cores.\n useTrainId set to {args.use_train_id} ')
    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )

    #'''
    mapped_categories = []
    print(id_map)
    for old_id, mapped_id in id_map.items():
        cat_data = categories[old_id]
        data_copy = cat_data.copy()
        data_copy['id'] = mapped_id
        data_copy['thingstuff_id'] = old_id
        mapped_categories.append(data_copy)
    save_json(mapped_categories, args.categories_stuff_panfpn_json)
    #'''

    print("Finished. time: {:.2f}s".format(time.time() - start))
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script converts panoptic COCO format to  the format required by the panoptic FPN. See this file's head for more information."
    )
    parser.add_argument('--panoptic_json_file', type=str,
                        help="JSON file with panoptic COCO format annotations")
    parser.add_argument('--panoptic_root', type=str,
                        help="Folder with panoptic COCO format panoptic segmentations")
    parser.add_argument('--sem_seg_root', type=str,
                        help="destination folder for the panoptic FPN semantic 'stuff' COCO format (semantic segmentations)",
                        default='./panoptic_stuff_panfpn')
    parser.add_argument('--categories_stuff_panfpn_json', type=str,
                        help="JSON containing the mapping from previous stuff categories to those used by the panoptic FPN ",
                        default='./categories_stuff_panfpn.json')


    parser.add_argument("--nproc", type=int,
                            default=1,
                            help="(default: 1) Number of processors to use for parallel computation")
    parser.add_argument("--use-train-id", type=str2bool, nargs='?',
                        const=True, default=True,
                            help="(default: True) Utilize categories which are cross-compatible with cityscapes and bdd100k")

    args = parser.parse_args()



    separate_coco_semantic_from_panoptic(args.panoptic_json_file, 
                                    args.panoptic_root, 
                                    args.sem_seg_root, 
                                    args)


  

