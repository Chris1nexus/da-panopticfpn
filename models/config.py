
from detectron2.config import CfgNode as CN
import os
from detectron2 import model_zoo

def add_da_panfpn_config(cfg):
    """
    Add config for TensorMask.
    """

    MODELS_DIRPATH = './configs'
    DETECTRON2_MODEL_FILENAME = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'
    MODEL_PATH = os.path.join(MODELS_DIRPATH, DETECTRON2_MODEL_FILENAME)
    cfg.merge_from_file(MODEL_PATH)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DETECTRON2_MODEL_FILENAME) 
    

    cfg.MODEL.BACKBONE.NAME ='build_resnet_multi_output_fpn_backbone'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =12 


    DA_CFG_PATH = os.path.join(MODELS_DIRPATH, 'da_cfg.yaml')
    cfg.DA = CN()
    cfg.DA.TRAIN = CN()
    cfg.DA.TEST = CN()
    cfg.DA.MODEL = CN()
    cfg.DA.TRAIN.SOURCE_DATASET_NAME ='cococarla_train_separated'#'cococarla_train_separated' #'bdd10k_train_separated'#'cococarla_train_separated'#'cococarla_train_separated'
    cfg.DA.TEST.SOURCE_DATASET_NAME = 'cococarla_val_separated'#'cococarla_val_separated'#'bdd10k_val_separated'#'cococarla_val_separated'#'cococarla_val_separated'
    cfg.DA.TRAIN.TARGET_DATASET_NAME = 'bdd10k_train_separated'#'cityscapes_train_separated'#'bdd10k_train_separated'
    cfg.DA.TEST.TARGET_DATASET_NAME = 'bdd10k_val_separated'#'cityscapes_val_separated'#'cityscapes_val_separated'#'bdd10k_val_separated'
    cfg.DA.TRAIN.OUT_OF_SAMPLE_DATASET_NAME = 'cityscapes_train_separated'
    cfg.DA.TEST.OUT_OF_SAMPLE_DATASET_NAME = 'cityscapes_val_separated'
    cfg.DA.MODEL.res2 = 0.#0.5#,0.125
    cfg.DA.MODEL.res3 = 0.#0.5#.125
    cfg.DA.MODEL.res4 = 0.#0.5#.125
    cfg.DA.MODEL.res5 = 0.#0.5#5#.125
    cfg.DA.MODEL.p2 = 0.#0.5#125 #0.5
    cfg.DA.MODEL.p3 = 0.#0.5
    cfg.DA.MODEL.p4 = 0.#0.5
    cfg.DA.MODEL.p5 = 0.#0.5
    cfg.DA.MODEL.p6 = 0. #0.5
    cfg.DA.MODEL.LAMBDA = 0. #0.1#1.
    cfg.DA.TRAIN.SOURCE_BATCH_SIZE = 4#8
    cfg.DA.TRAIN.TARGET_BATCH_SIZE = 4#8

    cfg.merge_from_file(DA_CFG_PATH)

    

    cfg.INPUT.FORMAT='RGB'
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATASETS.TRAIN = (cfg.DA.TRAIN.SOURCE_DATASET_NAME, )
    cfg.DATASETS.TEST = ( cfg.DA.TEST.SOURCE_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.TEST.EVAL_PERIOD = 100
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.MAX_ITER = 10000
    cfg.BASE_LR = 0.001
    cfg.WEIGHT_DECAY = 0.0
    cfg.WEIGHT_DECAY_NORM = 0.0
    cfg.WEIGHT_DECAY_BIAS = 0.0
    cfg.MAX_ITER = 10000
    
    cfg.SOLVER.IMS_PER_BATCH = cfg.DA.TRAIN.SOURCE_BATCH_SIZE #N_IMAGES_BATCH *2
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # POLY_ params are specific to WarmupPolyLR scheduler
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0

    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   


    cfg.LOG = CN()
    cfg.LOG.LOG_FOLDER = f'experiment_results_best{cfg.DA.TRAIN.SOURCE_DATASET_NAME.split("_")[0]  }_to_{cfg.DA.TRAIN.TARGET_DATASET_NAME.split("_")[0] }_test_{cfg.DA.TRAIN.OUT_OF_SAMPLE_DATASET_NAME.split("_")[0] }__'+\
            f'__roiheads-batchimg_{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}' +\
            f'__batch_{cfg.DA.TRAIN.SOURCE_BATCH_SIZE}_{cfg.DA.TRAIN.TARGET_BATCH_SIZE}__lr_{cfg.SOLVER.BASE_LR}'+\
            f'__res2_{cfg.DA.MODEL.res2}' + f'__res3_{cfg.DA.MODEL.res3}' + f'__res4_{cfg.DA.MODEL.res4}' + f'__res5_{cfg.DA.MODEL.res5}' +\
            f'__p2_{cfg.DA.MODEL.p2}' + f'__p3_{cfg.DA.MODEL.p3}' + f'__p4_{cfg.DA.MODEL.p4}' + f'__p5_{cfg.DA.MODEL.p5}' + f'__p6_{cfg.DA.MODEL.p6}' +\
            f'__sched_{cfg.SOLVER.LR_SCHEDULER_NAME}__opt_{cfg.SOLVER.OPTIMIZER}__lambda_{cfg.DA.MODEL.LAMBDA}/{MODEL_PATH.split("/")[-1].replace(".yaml","") }'

    os.makedirs(cfg.LOG.LOG_FOLDER, exist_ok=True)
