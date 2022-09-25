#!/usr/bin/env python
'''
This script converts detection COCO format to panoptic COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data.
'''
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

from panopticapi.utils import get_traceback, IdGenerator, save_json, rgb2id

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")



from labels import labels, id2label, trainId2label


@get_traceback
def convert_detection_to_panoptic_coco_format_single_core(
    proc_id, coco_detection, img_ids, categories, segmentations_folder, args
):
    

    annotations_panoptic = []
    for working_idx, img_id in enumerate(img_ids):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(img_ids)))
        id_generator = IdGenerator(categories)
        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        panoptic_record = {}
        panoptic_record['image_id'] = int(img_id)
        panoptic_file_name = img['file_name'].replace('.jpg', '.png')
        panoptic_record['file_name'] = panoptic_file_name
        segments_info = []

        # need this dictionary to store the position of stuff instances, since all 'stuff'
        # should be enclosed in the same segment_info item, regardless of how many separated parts of a 'stuff' are present in the current image annotation
        # This is needed to overcome the problem of a 'stuff' zone which is split in two parts by some blocking object 
        # (e.g. a pole splits the view on the road in two separate components), and the labeling of road 'stuff' possesses two instances of the road, BUT AS
        # STUFF, IT SHOULD BE UNCOUNTABLE, HENCE HAVE NO MORE THAN 1 'INSTANCE' IN THE PANOPTIC DATASET (meaning only one segment_info should be dedicated to each stuff category)
        # Quoting the panoptic task presentation paper:
        """
            When a pixel is labeled with li ∈ LSt,
            its corresponding instance id zi
            is irrelevant. That is, for
            stuff classes all pixels belong to the same instance (e.g., the
            same sky). 
        """
        # As such, this specification must be reflected both on the panoptic image and on the segment_info 

        stuff_category_to_idx = dict()

        if args.use_train_id:
            id2label_dict = trainId2label
        else:
            id2label_dict = id2label
        for ann_idx, ann in enumerate(anns):
            if ann['category_id'] not in categories:
                raise Exception('Panoptic coco categories file does not contain \
                    category with id: {}'.format(ann['category_id'])
                )

            label_info = id2label_dict[ann['category_id']]
            # sanity check
            if label_info.ignoreInEval:
                continue

            # note: segment_id is rgb2id(color), hence the inverse mapping holds when performign id2rgb(pan_format)
            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            # the same color is given to all stuff, so in the panoptic format all 'stuff' will have the same segment_id
            # we take care to ensure this behavior also in the segments_info, by employing the code in  if-else conditions below
            # inspired by bdd100k implementation of the panoptic mask creation 
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            pan_format[mask == 1] = color

            if ann['category_id'] not in stuff_category_to_idx:

                ann.pop('segmentation')
                ann.pop('image_id')
                ann['id'] = int(segment_id)
                ann['iscrowd'] = ann.get('iscrowd', 0) 
                area = np.sum(mask) 
                ann['area'] = int(area)
                segments_info.append(ann)
                
                if categories[ann['category_id']]['isthing'] == 0:
                    stuff_category_to_idx[ann['category_id']] = len(segments_info) - 1

            else:
                # this part of the if-else is only reached when ann == stuff
                segment_info = segments_info[stuff_category_to_idx[ann['category_id']]]

                # get merged mask of the current 'stuff' category(at this point, 
                # its old and current annotation are already merged in pan_format, thanks to line 100)
                pan_id_mask = rgb2id(pan_format)

                #merge bounding boxes
                # segmentid is always the same for stuff categories
                curr_mask = pan_id_mask == segment_id
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
                
                # update of the already existing segment_info
                segment_info['bbox'] = bbox
                segment_info["area"] = int(area) 
                segment_info["iscrowd"] = 0


        if np.sum(overlaps_map > 1) != 0:
            print("Segments for image {} overlap each other.".format(img_id))
        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, panoptic_file_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_ids)))
    return annotations_panoptic


def convert_detection_to_panoptic_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              args):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = output_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)




    print("CONVERTING...")
    print("COCO detection format:")
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO panoptic format")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(output_json_file))
    print('\n')

    coco_detection = COCO(input_json_file)
    img_ids = coco_detection.getImgIds()


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

    # ON DATASET LARGER THAN 150K Images, ram requirements become extreme for multiprocessing
    cpu_num = max(args.nproc, 1)
    img_ids_split = np.array_split(img_ids, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_ids_split[0])))

    if args.nproc > 1:
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, img_ids in enumerate(img_ids_split):
            p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core,
                                    (proc_id, coco_detection, img_ids, categories, segmentations_folder, args))
            processes.append(p)
        annotations_coco_panoptic = []
        for p in processes:
            annotations_coco_panoptic.extend(p.get())
    else:
        # avoid the use of workers if nproc == 1
        for proc_id, img_ids in enumerate(img_ids_split):
            annotations_coco_panoptic = convert_detection_to_panoptic_coco_format_single_core(proc_id, coco_detection, img_ids, categories, segmentations_folder, args)


    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    d_coco['annotations'] = annotations_coco_panoptic
    d_coco['categories'] = categories_list
    save_json(d_coco, output_json_file)

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
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--output_json_file', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if output_json_file is \
         X.json"
    )
    parser.add_argument("--nproc", type=int,
                            default=1,
                            help="(default: 1) Number of processors to use for parallel computation")
    parser.add_argument("--use-train-id", type=str2bool, nargs='?',
                        const=True, default=True,
                            help="(default: True) Utilize categories which are cross-compatible with cityscapes and bdd100k")

    args = parser.parse_args()
    convert_detection_to_panoptic_coco_format(args.input_json_file,
                                              args.segmentations_folder,
                                              args.output_json_file,
                                              args)