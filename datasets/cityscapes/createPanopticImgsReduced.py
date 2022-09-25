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


# The main method
def convert2panoptic(cityscapesPath=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if cityscapesPath is None:
        if 'CITYSCAPES_DATASET' in os.environ:
            cityscapesPath = os.environ['CITYSCAPES_DATASET']
        else:
            cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
        cityscapesPath = os.path.join(cityscapesPath, "gtFine")

    if outputFolder is None:
        outputFolder = cityscapesPath

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
        outputBaseFile = "cityscapes_panoptic_{}{}".format(setName, trainIfSuffix)
        outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
        print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
        panopticFolder = os.path.join(outputFolder, outputBaseFile)
        if not os.path.isdir(panopticFolder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
            os.makedirs(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        images = []
        annotations = []
        # keep track of how many images, s.t. image_id can be set to an integer value 
        image_idx = 0        
        for progress, f in enumerate(files):


            id_generator = IdGenerator(categories)


            originalFormat = np.array(Image.open(f))

            fileName = os.path.basename(f)
            imageId = fileName.replace("_gtFine_instanceIds.png", "")
            inputFileName = fileName.replace("_instanceIds.png", "_leftImg8bit.jpg")
            outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")
            # image entry, id for image is its filename without extension
            images.append({"id": image_idx,
                           "width": int(originalFormat.shape[1]),
                           "height": int(originalFormat.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            segmentIds = np.unique(originalFormat)
            segmInfo = []
            for segmentId in segmentIds:
                if segmentId < 1000:
                    semanticId = segmentId
                    isCrowd = 1
                else:
                    semanticId = segmentId // 1000
                    isCrowd = 0
                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval:
                    continue
                if not labelInfo.hasInstances:
                    isCrowd = 0



                mask = originalFormat == segmentId
                segment_id, color = id_generator.get_id_and_color(categoryId)
                pan_format[mask] = color

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [float(x), float(y), float(width), float(height)]

                segmInfo.append({"id": int(segment_id),#int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': image_idx, #imageId,
                                'file_name': outputFileName,
                                "segments_info": segmInfo})
            image_idx += 1
            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()

        print("\nSaving the json file {}".format(outFile))
        d = {'images': images,
             'annotations': annotations,
             'categories': categories_list}#categories}
        with open(outFile, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


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
    print(f'TrainId: {args.useTrainId}')
    convert2panoptic(args.cityscapesPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
