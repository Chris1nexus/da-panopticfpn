



from detectron2.engine import DefaultTrainer
import cv2
import logging
import json
import os
import scipy.misc
import numpy as np

import random

from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import default_argument_parser, default_setup, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances



    

from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated,register_coco_panoptic

from tqdm import tqdm




def load_things_metadata(meta, cats):

    thing_cats = [c for c in sorted(cats, key=lambda x: x["thingstuff_id"]) if c['isthing']==1 ]
    thing_classes = [c["name"] for idx, c in enumerate(thing_cats) ]
    thing_id_to_contiguous_id = {c['thingstuff_id']: idx  for idx, c in enumerate(thing_cats) }
    thing_colors =  [c["color"] for idx, c in enumerate(thing_cats) ]

    
    meta.thing_classes = thing_classes
    meta.thing_colors = thing_colors

def register_datasets(args):

 
    target1_train_name='cityscapes_train'
    target1_train_metadata = {}
    target1_train_image_root= os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/images/train')
    target1_train_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_panoptic_train_trainId')
    target1_train_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_panoptic_train_trainId.json')
    target1_train_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/train_sem_stuff')
    target1_train_instances_json = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_instances_train_trainId.json')
    register_coco_panoptic_separated( target1_train_name, target1_train_metadata, target1_train_image_root, 
                            target1_train_panoptic_root, target1_train_panoptic_json, 
                            target1_train_sem_seg_root, target1_train_instances_json)



    target1_val_name='cityscapes_val'
    target1_val_metadata = {}
    target1_val_image_root= os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/images/val')
    target1_val_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_panoptic_val_trainId')
    target1_val_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_panoptic_val_trainId.json')
    target1_val_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/val_sem_stuff')
    target1_val_instances_json = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/cityscapes_instances_val_trainId.json')
    register_coco_panoptic_separated( target1_val_name, target1_val_metadata, target1_val_image_root, 
                            target1_val_panoptic_root, target1_val_panoptic_json, 
                            target1_val_sem_seg_root, target1_val_instances_json)

    stuff_info_path = os.path.join(os.environ['DATA_ROOT'],'pan_cityscapes_reduced/categories_stuff_panfpn.json')
    with open(stuff_info_path)  as f:
        stuff_categories = json.load(f)
    

    stuff_dataset_id_to_contiguous_id = {cat_dict['thingstuff_id']: cat_dict['id']  for cat_dict in stuff_categories  }
    stuff_cont_dict = {cat_dict['id']: cat_dict['name']  for cat_dict in stuff_categories  }
    stuff_cont_dict[0] = 'thing'
    stuff_classes = list(map(lambda x:x[1], sorted(stuff_cont_dict.items(), key=lambda x: x[0])  ))
    stuff_colors_dict = {cat_dict['id']: cat_dict['color']  for cat_dict in stuff_categories  }
    stuff_colors = list(map(lambda item:  (item[1][0],item[1][1], item[1][2]), sorted(stuff_colors_dict.items(), key=lambda x: x[0])  ))

    stuff_names_dict = {cat_dict['id']: cat_dict['name']  for cat_dict in stuff_categories  }
    stuff_names = list(map(lambda x:  x[1]  , sorted(stuff_names_dict.items(), key=lambda x: x[0])  ))



    load_things_metadata(MetadataCatalog.get(target1_train_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(target1_train_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(target1_train_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(target1_train_name+'_separated')\
        .stuff_colors = stuff_colors
    load_things_metadata(MetadataCatalog.get(target1_val_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(target1_val_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(target1_val_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(target1_val_name+'_separated')\
        .stuff_colors = stuff_colors



    target2_train_name='bdd10k_train'
    target2_train_metadata = {}
    target2_train_image_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_images/images/10k/train')
    target2_train_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_panoptic_reduced/train')
    target2_train_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_panoptic_reduced_train.json')
    target2_train_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/train_sem_stuff')
    target2_train_instances_json = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_instances_reduced_train.json')
    register_coco_panoptic_separated( target2_train_name, target2_train_metadata, target2_train_image_root, 
                            target2_train_panoptic_root, target2_train_panoptic_json, 
                            target2_train_sem_seg_root, target2_train_instances_json)




    target2_val_name='bdd10k_val'
    target2_val_metadata = {}
    target2_val_image_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_images/images/10k/val')
    target2_val_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_panoptic_reduced/val')
    target2_val_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_panoptic_reduced_val.json')
    target2_val_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/val_sem_stuff')
    target2_val_instances_json = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/bdd100k_instances_reduced_val.json')
    register_coco_panoptic_separated( target2_val_name, target2_val_metadata, target2_val_image_root, 
                            target2_val_panoptic_root, target2_val_panoptic_json, 
                            target2_val_sem_seg_root, target2_val_instances_json)

    stuff_info_path = os.path.join(os.environ['DATA_ROOT'],'bdd100k_reduced/categories_stuff_panfpn.json')
    with open(stuff_info_path)  as f:
        stuff_categories = json.load(f)
        
    stuff_dataset_id_to_contiguous_id = {cat_dict['thingstuff_id']: cat_dict['id']  for cat_dict in stuff_categories  }
    stuff_cont_dict = {cat_dict['id']: cat_dict['name']  for cat_dict in stuff_categories  }
    stuff_cont_dict[0] = 'thing'
    stuff_classes = list(map(lambda x:x[1], sorted(stuff_cont_dict.items(), key=lambda x: x[0])  ))
    stuff_colors_dict = {cat_dict['id']: cat_dict['color']  for cat_dict in stuff_categories  }
    stuff_colors = list(map(lambda item:  (item[1][0],item[1][1], item[1][2]), sorted(stuff_colors_dict.items(), key=lambda x: x[0])  ))

    load_things_metadata(MetadataCatalog.get(target2_train_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(target2_train_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(target2_train_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(target2_train_name+'_separated')\
        .stuff_colors = stuff_colors

    load_things_metadata(MetadataCatalog.get(target2_val_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(target2_val_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(target2_val_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(target2_val_name+'_separated')\
        .stuff_colors = stuff_colors


    syn_train_name='cococarla_train'
    syn_train_metadata = {}
    syn_train_image_root = os.path.join(os.environ['DATA_ROOT'],'coco_carla/train_images')
    syn_train_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'coco_carla/train_panoptic')
    syn_train_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'coco_carla/train_annotations/panoptic.json')
    syn_train_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'coco_carla/train_sem_stuff')
    syn_train_instances_json = os.path.join(os.environ['DATA_ROOT'],'coco_carla/train_annotations/instances_things.json')
    register_coco_panoptic_separated( syn_train_name, syn_train_metadata, syn_train_image_root, 
                            syn_train_panoptic_root, syn_train_panoptic_json, 
                            syn_train_sem_seg_root, syn_train_instances_json)



    syn_val_name='cococarla_val'
    syn_val_metadata = {}
    syn_val_image_root= os.path.join(os.environ['DATA_ROOT'],'coco_carla/val_images')
    syn_val_panoptic_root = os.path.join(os.environ['DATA_ROOT'],'coco_carla/val_panoptic')
    syn_val_panoptic_json = os.path.join(os.environ['DATA_ROOT'],'coco_carla/val_annotations/panoptic.json')
    syn_val_sem_seg_root = os.path.join(os.environ['DATA_ROOT'],'coco_carla/val_sem_stuff')
    syn_val_instances_json = os.path.join(os.environ['DATA_ROOT'],'coco_carla/val_annotations/instances_things.json')
    register_coco_panoptic_separated( syn_val_name, syn_val_metadata, syn_val_image_root, 
                            syn_val_panoptic_root, syn_val_panoptic_json, 
                            syn_val_sem_seg_root, syn_val_instances_json)
    stuff_info_path = os.path.join(os.environ['DATA_ROOT'],'coco_carla/categories_stuff_panfpn.json') 
    with open(stuff_info_path)  as f:
        stuff_categories = json.load(f)
        
    #stuff_dataset_id_to_contiguous_id = {cat_dict['thingstuff_id']: cat_dict['id']  for cat_dict in stuff_categories if cat_dict['id']!= 0 and cat_dict['id'] != 255 }
    stuff_dataset_id_to_contiguous_id = {cat_dict['thingstuff_id']: cat_dict['id']  for cat_dict in stuff_categories  }
    stuff_cont_dict = {cat_dict['id']: cat_dict['name']  for cat_dict in stuff_categories  }
    stuff_cont_dict[0] = 'thing'
    stuff_classes = list(map(lambda x:x[1], sorted(stuff_cont_dict.items(), key=lambda x: x[0])  ))
    stuff_colors_dict = {cat_dict['id']: cat_dict['color']  for cat_dict in stuff_categories  }
    stuff_colors = list(map(lambda item:  (item[1][0],item[1][1], item[1][2]), sorted(stuff_colors_dict.items(), key=lambda x: x[0])  ))

    load_things_metadata(MetadataCatalog.get(syn_train_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(syn_train_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(syn_train_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(syn_train_name+'_separated')\
        .stuff_colors = stuff_colors

    load_things_metadata(MetadataCatalog.get(syn_val_name+'_separated'), 
        stuff_categories)
    MetadataCatalog.get(syn_val_name+'_separated')\
        .stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get(syn_val_name+'_separated')\
        .stuff_classes = stuff_classes
    MetadataCatalog.get(syn_val_name+'_separated')\
        .stuff_colors = stuff_colors
        

