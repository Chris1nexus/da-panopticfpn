"""Convert BDD100K bitmasks to COCO panoptic segmentation format."""

import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
# requires scalabel install from setup.py clone of the official github repo, otherwise many datatypes are left undefined
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayI32
from scalabel.label.coco_typing import (
    ImgType,
    PanopticAnnType,
    PanopticCatType,
    PanopticGtType,
    PanopticSegType,
)
from tqdm import tqdm

from bdd100k.common.logger import logger
from bdd100k.common.utils import list_files
from label_reduced import labels, trainId2label, id2label
#from .to_coco import bitmasks_loader
#from .to_mask import STUFF_NUM
from bdd100k.common.typing import BDD100KConfig, InstanceType
from scalabel.label.typing import Box2D,ImageSize # Box2D accepts floats

from panopticapi.utils import IdGenerator, rgb2id
import functools


def box2d_to_bbox(box_2d: Box2D) -> List[float]:
    """Convert Scalabel Box2D into COCO bbox."""
    width = box_2d.x2 - box_2d.x1 + 1
    height = box_2d.y2 - box_2d.y1 + 1
    return [box_2d.x1, box_2d.y1, width, height]

'''
MODIFIED METHOD, IN ORDER TO UTILIZE: FLOAT BOUNDING BOX ABSOLUTE COORDINATES
'''
def mask_to_box2d(mask: np.ndarray) -> Box2D:
    """Convert mask into Box2D."""
    x_inds = np.nonzero(np.sum(mask, axis=0))[0]
    y_inds = np.nonzero(np.sum(mask, axis=1))[0]
    #x1, x2 = int(np.min(x_inds)), int(np.max(x_inds))
    #y1, y2 = int(np.min(y_inds)), int(np.max(y_inds))
    x1, x2 = np.min(x_inds), np.max(x_inds)
    y1, y2 = np.min(y_inds), np.max(y_inds)
    box_2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)
    return box_2d


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """Convert mask into bbox."""
    box_2d = mask_to_box2d(mask)
    bbox = box2d_to_bbox(box_2d)
    return bbox
def bitmasks_loader(mask_name: str) -> Tuple[List[InstanceType], ImageSize]:
    """Parse instances from the bitmask."""
    if mask_name.endswith(".jpg"):
        mask_name = mask_name.replace(".jpg", ".png")
    bitmask = np.asarray(Image.open(mask_name), dtype=np.int32)
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]
    indentity_map = (
        (category_map << 24) + (attributes_map << 16) + instance_map
    )

    instances: List[InstanceType] = []

    identities = np.unique(indentity_map)
    for identity in identities:
        mask = np.equal(indentity_map, identity)
        category_id = (identity >> 24) & 255
        attribute = (identity >> 16) & 255
        instance_id = identity & 65535
        if category_id == 0:
            continue

        bbox = mask_to_bbox(mask)
        area = np.sum(mask).tolist()

        instance = InstanceType(
            instance_id=int(instance_id),
            category_id=int(category_id),
            truncated=bool(attribute & (1 << 3)),
            occluded=bool(attribute & (1 << 2)),
            crowd=bool(attribute & (1 << 1)),
            ignored=bool(attribute & (1 << 0)),
            mask=mask,
            bbox=bbox,
            area=area,
        )
        instances.append(instance)

    instances = sorted(instances, key=lambda instance: instance["instance_id"])
    img_shape = ImageSize(height=bitmask.shape[0], width=bitmask.shape[1])

    return (instances, img_shape)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="bdd100k bitmasks to coco panoptic format"
    )
    parser.add_argument("-i", "--input", help="path to the bitmask folder.")
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco panoptic formatted json file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "-pb",
        "--pan-mask-base",
        help="Path to the output panoptic segmentation mask folder.",
    )

    parser.add_argument("--use-train-id", type=str2bool, nargs='?',
                        const=True, default=True,
                            help="(default: True) Utilize categories which are cross-compatible with cityscapes and bdd100k")
    return parser.parse_args()



def bitmask2pan_both(image: ImgType, mask_name: str, pan_name: str, use_train_id :bool)-> Tuple[ImgType, PanopticAnnType]:
    

    
    instances, img_shape = bitmasks_loader(mask_name)
    image["height"] = img_shape.height
    image["width"] = img_shape.width

    pan_fmt = np.zeros((img_shape.height, img_shape.width, 3), dtype=np.uint8)


    cat_id_to_idx: Dict[int, int] = {}
    

    id2label_dict = trainId2label if use_train_id else id2label
    categories_list = []
    for id, label_info in id2label_dict.items(): 
        if label_info.ignoreInEval:
                continue
        categories_list.append({'id': int(label_info.trainId) if use_train_id else int(label_info.id),
                               'name': label_info.name,
                               'color': label_info.color,
                               'supercategory': label_info.category,
                               'isthing': 1 if label_info.hasInstances else 0})
    categories = {category['id']: category for category in categories_list}
    id_generator = IdGenerator(categories ) 
    
    
    segments_info: List[PanopticSegType] = []
    for instance in instances:
        # original catid
        category_id = instance["category_id"]
        # this category_id comes from the bdd100k id set, hence
        # to get its infos we have to use the id2label
        label_info = id2label[category_id]#id2label_dict[category_id]
        if label_info.ignoreInEval:
                continue
        selected_category_id = label_info.trainId if use_train_id else category_id
        ignore = label_info.ignoreInEval if use_train_id else 0

        segment_id, color = id_generator.get_id_and_color(selected_category_id)
        
        mask = instance['mask']
        pan_fmt[mask] = color
        
        if selected_category_id not in cat_id_to_idx:#category_id not in cat_id_to_idx:            
    
            
            segment_info = PanopticSegType(
                id=segment_id, #instance["instance_id"],
                category_id=selected_category_id , #category_id,
                area=instance["area"],
                iscrowd=instance["crowd"] or instance["ignored"],
                ignore=ignore #0,
            )
            segments_info.append(segment_info)
            
            
            if label_info.hasInstances == False: #category_id <= STUFF_NUM:
                #cat_id_to_idx[category_id] = len(segment_info) - 1
                cat_id_to_idx[selected_category_id] = len(segments_info) - 1
        else:
            segment_info = segments_info[cat_id_to_idx[selected_category_id]]#category_id]]
            
            pan_id_mask = rgb2id(pan_fmt)
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
            
            segment_info["area"] = int(area) # += instance["area"]
            segment_info["iscrowd"] = 0
    annotation = PanopticAnnType(
        image_id=image["id"],
        file_name=image["file_name"].replace(".jpg", ".png"),
        segments_info=segments_info,
    )
    pan_mask = Image.fromarray(pan_fmt)
    pan_mask.save(pan_name)

    return image, annotation



def bitmask2panseg_parallel(
    mask_base: str,
    pan_mask_base: str,
    images: List[ImgType],
    use_train_id: bool,
    nproc: int = NPROC,
) -> Tuple[List[ImgType],List[PanopticAnnType]]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    mask_names = [
        os.path.join(mask_base, image["file_name"]) for image in images
    ]
    pan_names = [
        os.path.join(pan_mask_base, image["file_name"]) for image in images
    ]

    with Pool(nproc) as pool:
        images, annotations = zip(*pool.starmap(functools.partial(bitmask2pan_both, use_train_id=use_train_id),
            tqdm(zip(images, mask_names, pan_names), total=len(images)),
        ))


    images = sorted(images, key=lambda img: img["id"])
    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    return images, annotations


def bitmask2coco_pan_seg(
    mask_base: str,
    pan_mask_base: str,
    use_train_id: bool,
    nproc: int = NPROC,
) -> PanopticGtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    if not os.path.exists(pan_mask_base):
        os.makedirs(pan_mask_base)

    files = list_files(mask_base, suffix=".png")
    images: List[ImgType] = []

    logger.info("Collecting bitmasks...")

    image_id = 0
    for file_ in tqdm(files):
        image_id += 1
        image = ImgType(id=image_id, file_name=file_.replace('.png', '.jpg') )
        images.append(image)

    images, annotations = bitmask2panseg_parallel(
        mask_base, pan_mask_base, images, use_train_id, nproc
    )
    categories: List[PanopticCatType] = [
        PanopticCatType(
            id=int(label.trainId) if use_train_id else int(label.id),
            name=label.name,
            supercategory=label.category,
            isthing=label.hasInstances,
            color=label.color,
        )
        for label in ( trainId2label.values() if use_train_id else id2label.values())   \
        if label.ignoreInEval == False
    ]
    return PanopticGtType(
        categories=categories,
        images=images,
        annotations=annotations,
    )


def main() -> None:
    """Main function."""
    args = parse_args()

    logger.info("Start format converting...")
    coco = bitmask2coco_pan_seg(args.input, args.pan_mask_base, args.use_train_id, args.nproc)
    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(coco, fp)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
