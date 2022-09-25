from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, IdGenerator, save_json

import cv2
import pandas as pd
from ast import literal_eval as make_tuple

import glob
import os
import cv2
import numpy as np
from panopticapi.utils import rgb2id
from pycocotools.mask import encode, decode





from labels import labels, id2label, trainId2label




def instance_image_to_coco_detection(proc_id, 
                                    paths, 
                                    datasets_directory,
                                    image_dirpath, 
                                    split_block_size,
                                    args):

    images = []
    annotations_by_image = []


    for working_idx, instance_path in enumerate(paths):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(paths)))

        image_id = proc_id*split_block_size + working_idx
        image_path = instance_path.replace('instance_segmentation', 'rgb')
        img_filename = '--'.join( [x for x in image_path.replace(datasets_directory, "").split('/')  if len(x) > 0]  )\
                .replace('.png', '.jpg')


        inst_img = cv2.imread(instance_path)[:,:,::-1]
        # image in RGB format (s.t. it is compatible with panopticapi rgb2id specifications)
        id_mask = rgb2id(inst_img)
        semantic_mask = inst_img[:,:,0]
        num_instances = np.unique(id_mask).shape[0]

        dest_image_filepath = os.path.join(image_dirpath, img_filename)

        im = Image.open(image_path)
        rgb_im = im.convert('RGB')
        rgb_im.save(dest_image_filepath)


        images.append({"id": image_id,
                            "width": inst_img.shape[1],
                            "height": inst_img.shape[0],
                            "file_name": img_filename
                               })

        coco_annot_dicts = []        
        # ascending id order
        for instance_id in np.unique(id_mask):


            semantic_query = semantic_mask[id_mask == instance_id]
            found_tags = np.unique(semantic_query)
            if len(found_tags) > 1:
                query_ids, query_counts = np.unique(semantic_query, return_counts=True)
                most_frequent_sem_id = sorted( (zip(query_ids, query_counts)), key=lambda x:x[1], reverse=True)[0][0]
                semantic_id = most_frequent_sem_id
            else:
                semantic_id = found_tags[0]
            semantic_id = semantic_id.item()
            

            # from id used in the dataset to the abstraction level of the machine learning dataset ID
            label_info = id2label[semantic_id]
            semantic_id = label_info.trainId if args.use_train_id else label_info.id
            if label_info.ignoreInEval:
                    continue

            # if it is a 'stuff' item, skip and continue to the next instance id 
            if args.panoptic_only == False:
                # skip current instance, if things only and found a stuff category (label_info.hasInstances == 0 when it is a 'stuff')
                if args.things_only and label_info.hasInstances == 0:
                    continue
                # skip current instance, if stuff only and found a thing category (label_info.hasInstances == 1 when it is a 'thing')
                if args.stuff_only and label_info.hasInstances == 1:
                    continue

            
            curr_mask = id_mask == instance_id

            rle_polygon = encode(np.asfortranarray(curr_mask))
            rle_polygon['counts'] = rle_polygon['counts'].decode()
            
            area = np.sum(curr_mask) # segment area computation
            # bbox computation for a segment
            hor = np.sum(curr_mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(curr_mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [float(x), float(y), float(width), float(height) ]
            
            is_crowd = 0
            



            coco_annot_dicts.append({
                            "image_id": int(image_id),
                            "bbox":bbox,
                            "area":int(area),
                            "iscrowd":0,
                            "segmentation": rle_polygon,
                            "category_id": int(semantic_id),
                        })
        annotations_by_image.append(coco_annot_dicts)
    return images, annotations_by_image



def convert_to_detection(datasets_directory,
                                        image_dirpath,
                                        annot_dirpath,
                                        output_json_file,
                                        args):
    start_time = time.time()


    if not os.path.exists(image_dirpath):
        os.makedirs(image_dirpath)
    if not os.path.exists(annot_dirpath):
        os.makedirs(annot_dirpath)

    print("CONVERTING...")
    print("COCO detection format:")

    print("TO")
    print("COCO panoptic format")

    print('\n')

    
    #'../datasets/*/sensor_data/*/instance_segmentation/*.png
    all_paths = glob.glob( os.path.join(datasets_directory, '*', 'sensor_data','*','instance_segmentation','*.png')   )


    cpu_num = multiprocessing.cpu_count()
    img_paths_split = np.array_split(all_paths, cpu_num)
    split_block_size = len(img_paths_split[0])
    print("Number of cores: {}, images per core: {}".format(cpu_num, split_block_size ))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_paths in enumerate(img_paths_split):
        p = workers.apply_async(instance_image_to_coco_detection,
                                (proc_id, img_paths, 
                                    datasets_directory,
                                    image_dirpath, 
                                    split_block_size,
                                    args))
        processes.append(p)

    images_list =[]
    annotations_group_list =[]
    for p in processes:
        images, annotations = p.get()
        
        images_list.extend(images)
        annotations_group_list.extend(annotations)
    
    annot_idx = 0
    annotations_list = []
    for annot_group in annotations_group_list:
        for ann in annot_group:
            ann['id'] = annot_idx
            annotations_list.append(ann)
            annot_idx += 1



    # The following lines are used in order to resolve conflicts that arise with the usage of the simple labels list .
    # See labels.py to notice that trainId2label reverses the labels list, so that subcategories are grouped into the super categories, 
    # such that when using trainId, only the semantically larger categories are utilized.
    # E.g. dynamic objects are treated as VOID, id=22, trainId=255, by this convention, it is grouped in the unlabeled category with id=0 and trainId=255
    categories_list = []
    if args.use_train_id:
        id2label_dict = trainId2label
    else:
        id2label_dict = id2label

    for id, label_info in id2label_dict.items(): #trainId2label.items()
        if label_info.ignoreInEval:
                continue
        categories_list.append({'id': int(label_info.trainId) if args.use_train_id else int(label_info.id),
                               'name': label_info.name,
                               'color': label_info.color,
                               'supercategory': label_info.category,
                               'isthing': 1 if label_info.hasInstances else 0})


    if args.things_only:
        categories_list = [ cat for cat in categories_list if cat['isthing'] == 1]
    if args.stuff_only:
         categories_list = [ cat for cat in categories_list if cat['isthing'] == 0]
    d = {'images': images_list,
         'annotations': annotations_list,
         'categories': categories_list,
        }

    out_json_path = os.path.join(annot_dirpath, output_json_file)
    save_json(d, out_json_path)


    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

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
        description="This script converts detection COCO format to panoptic \
            COCO format. See this file's head for more information."
    )
    parser.add_argument('--datadir', type=str,
                        help="directory containing one or more carla datasets")

    parser.add_argument('--image-dirpath', type=str,
                        help="directory containing images")
    parser.add_argument('--annot-dirpath', type=str,
                        help="directory containing annotations")
    parser.add_argument('--output-json-file', type=str,
                        help="JSON file with panoptic COCO format",
                        default='./instances_coco_carla.json')
    parser.add_argument("--use-train-id", type=str2bool, nargs='?',
                        const=True, default=True,
                            help="(default: True) Utilize categories which are cross-compatible with cityscapes and bdd100k")
   
    parser.add_argument("--things-only", type=str2bool, nargs='?',
                        const=True, default=False,
                            help="(default: False) generate instance only annotation file (COCO instances.json).\nRequired for COCO instance detection (COCO-detection challenge)\n Stuff categories are ignored and skipped")
    parser.add_argument("--stuff-only", type=str2bool, nargs='?',
                        const=True, default=False,
                             help="(default: False) generate stuff only annotation file (COCO stuff.json).\nRequired for COCO stuff detection (COCO-stuff challenge)\n Thing categories are ignored and skipped")

    parser.add_argument("--panoptic-only", type=str2bool, nargs='?',
                        const=True, default=False,
                            help="(default: False) generate panoptic annotation file (COCO panoptic.json).\nRequired for COCO panoptic detection (COCO-panoptic challenge)\n Both thing and stuff are considered, but this file serves only as input to the argument --input_json_file in the PANOPTICAPI converters/detection2panoptic_coco_format.py \n see the link for more details: https://github.com/cocodataset/panopticapi/blob/master/CONVERTERS.md ")

      
    args = parser.parse_args()
    assert (args.things_only ^ args.stuff_only ^ args.panoptic_only ) &  (not (args.things_only & args.stuff_only & args.panoptic_only )), \
            f"Error exactly one of things_only, stuff_only, panoptic_only must be true, but found:\nthings_only:{args.things_only}\nstuff_only:{args.stuff_only}\npanoptic_only:{args.panoptic_only }"
  

    convert_to_detection(args.datadir,
                                                args.image_dirpath,
                                                args.annot_dirpath,
                                              args.output_json_file,
                                            args)