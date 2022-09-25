#!/usr/bin/python
#
# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
# The convertion is working for 'fine' set of the annotations.
#
# By default with this tool uses IDs specified in labels.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For cityscapes image_id has form <city>_123456_123456 and corresponds to the prefix
# of cityscapes image files.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from labels_reduced import id2label, labels, trainId2label
from panopticapi.utils import IdGenerator
from pycocotools.mask import encode, decode

# The main method
def convert2COCOinstances(cityscapesPath=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if cityscapesPath is None:
        if 'CITYSCAPES_DATASET' in os.environ:
            cityscapesPath = os.environ['CITYSCAPES_DATASET']
        else:
            cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
        cityscapesPath = os.path.join(cityscapesPath, "gtFine")

    if outputFolder is None:
        outputFolder = cityscapesPath
    elif not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    categories_list = []
    # define which set of ids to use for the panoptic task
    if useTrainId:
        id2label_dict = trainId2label
    else:
        id2label_dict = id2label
    # parse id set from cityscapes like to the one used by coco environments
    for id, label_info in id2label_dict.items(): 
        if label_info.ignoreInEval:
                continue
        if label_info.hasInstances == False:
                continue
        categories_list.append({'id': int(label_info.trainId) if useTrainId else int(label_info.id),
                               'name': label_info.name,
                               'color': label_info.color,
                               'supercategory': label_info.category,
                               'isthing': 1 if label_info.hasInstances else 0})
    categories = {category['id']: category for category in categories_list}

    for setName in setNames:
        # how to search for all ground truth
        searchFine   = os.path.join(cityscapesPath, setName, "*", "*_instanceIds.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        trainIfSuffix = "_trainId" if useTrainId else ""
        outputBaseFile = "cityscapes_instances_{}{}".format(setName, trainIfSuffix)
        outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
        print("Json file with the annotations in COCO format will be saved in {}".format(outFile))

        # keep track of how many annotations, such that each can be assigned a unique ID for the COCO format
        annot_idx = 0
        # keep track of how many images, s.t. image_id can be set to an integer value 
        image_idx = 0
        images = []
        annotations = []
        for progress, f in enumerate(files):

            id_mask = np.array(Image.open(f))

            fileName = os.path.basename(f)
            image_id = fileName.replace("_gtFine_instanceIds.png", "")
            inputFileName = fileName.replace("_instanceIds.png", "_leftImg8bit.jpg")
            #outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")
            # image entry, id for image is its filename without extension

            
            images.append({"id": image_idx,#image_id,
                           "width": int(id_mask.shape[1]),
                           "height": int(id_mask.shape[0]),
                           "file_name": inputFileName})

      
            coco_annot_dicts = []        
            for instance_id in np.unique(id_mask):

                if instance_id < 1000:
                    semantic_id = instance_id
                else:
                    semantic_id = instance_id //1000
                # get metadata of the current category
                label_info = id2label[semantic_id]
                if label_info.ignoreInEval:
                    continue
                if label_info.hasInstances == False:
                    continue
                # extract proper id from current semantic category
                category_id = label_info.trainId if useTrainId else label_info.id
                
                # get current instance mask
                curr_id_mask = instance_id == id_mask
                # extract rle polygon
                rle_polygon = encode(np.asfortranarray(curr_id_mask))
                rle_polygon['counts'] = rle_polygon['counts'].decode()
                
                # bounding box and area computation
                area = np.sum(curr_id_mask) # segment area computation
                hor = np.sum(curr_id_mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(curr_id_mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [float(x), float(y), float(width), float(height)]


                coco_annot_dicts.append({
                            "id" : annot_idx,
                            "image_id": image_idx,
                            "bbox":bbox,
                            "area":int(area),
                            "iscrowd": 0,
                            "segmentation": rle_polygon,
                            "category_id": int(category_id),
                        })
                annot_idx += 1

            annotations.extend(coco_annot_dicts)
            image_idx += 1

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()

        print("\nSaving the json file {}".format(outFile))


        d = {'images': images,
             'annotations': annotations,
             'categories': categories_list}
        with open(outFile, 'w') as f:
            json.dump(d, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="cityscapesPath",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train", "test"],
                        type=str)
    args = parser.parse_args()

    convert2COCOinstances(args.cityscapesPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()



