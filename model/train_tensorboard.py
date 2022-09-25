import logging
import os
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer, Checkpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated,register_coco_panoptic
import detectron2.modeling.meta_arch
import detectron2.modeling.backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import build_model
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter

from PIL import Image
import torchvision
from IPython.display import clear_output, display 
import copy





from models import add_da_panfpn_config
from models.da_panfpn import DAPanopticFPN
from datasets import register_datasets
from panopticapi.utils import save_json
from models.lr_scheduler import WarmupPolyLR
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler
from detectron2.config import CfgNode
logger = logging.getLogger("detectron2")


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        )
    else:
        return build_d2_lr_scheduler(cfg, optimizer)



def get_pred(model, batch, dataset_metadata, step=4,plot=False):
    processed_preds = []

    for idx in range(0,len(batch), step):
        sub_batch = batch[idx:(idx+step)]
        preds = model(sub_batch)
        
        for img_data, pred in zip(sub_batch, preds):

            v = Visualizer( img_data['image'].numpy().transpose((1,2,0)), #im,
                           metadata=dataset_metadata, 
                           scale=1, 
                           instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            pan_seg, seg_info = pred['panoptic_seg']
            out = v.draw_panoptic_seg(pan_seg.to('cpu'), 
                                      seg_info, area_threshold=None, alpha=0.7)
            if plot:
                display(Image.fromarray(out.get_image())  )
            processed_preds.append(out.get_image())
        
    return processed_preds

def fill_batch(dataloader, batch_size=8):
    iterator = iter(dataloader) 
    batch = []
    while len(batch) < batch_size:
        curr_batch = next(iterator)
        curr_batch_copy = copy.deepcopy(curr_batch)  
        del curr_batch
        batch.extend(curr_batch_copy  )
    del iterator
    return batch[:batch_size]






def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    #if evaluator_type in ["coco", "coco_panoptic_seg"]:
    #    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results



def do_train(cfg_source, cfg_target1, cfg_target2, model, resume = False):


    model.train()

    print(f'MAXIMUM NUMBER OF ITERATIONS: {cfg_source.SOLVER.MAX_ITER}')
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)



    checkpointer = DetectionCheckpointer(
        model, cfg_source.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg_source.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg_source.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    i = 1

    min_data_len = min(len(DatasetCatalog.get(cfg_source.DATASETS.TRAIN[0])), 
                                len(DatasetCatalog.get(cfg_source.DATASETS.TRAIN[0])) )#cfg_target.DATASETS.TRAIN[0])) )
    max_epoch = max_iter / min_data_len
    current_epoch = 0
    data_len = min_data_len


    lambda_ = cfg_source.DA.MODEL.LAMBDA

    # upper bounds to the alpha values

    ub_alpha_dict = {
        'res2': cfg_source.DA.MODEL.res2,
        'res3': cfg_source.DA.MODEL.res3,
        'res4': cfg_source.DA.MODEL.res4,
        'res5': cfg_source.DA.MODEL.res5,
        'p2': cfg_source.DA.MODEL.p2,
        'p3': cfg_source.DA.MODEL.p3,
        'p4': cfg_source.DA.MODEL.p4,
        'p5': cfg_source.DA.MODEL.p5,
        'p6': cfg_source.DA.MODEL.p6,
    }
    train_alpha_dict = ub_alpha_dict.copy()


    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_target1 = build_detection_train_loader(cfg_target1)
    data_loader_target2 = build_detection_train_loader(cfg_target2)

    # although these are test dataloaders, these have to be created with  build_detection_train_loader
    # otherwise this would not yield batches needed for computing validation 
    # (detectron2 creates single sample batches when creating test dataloader)
    # need batches with the same format to the train ones to compute the val loss
    cfg_source_val = copy.deepcopy(cfg_source)
    cfg_target1_val = copy.deepcopy(cfg_target1)
    cfg_target2_val = copy.deepcopy(cfg_target2)
    cfg_source_val.DATASETS.TRAIN = cfg_source_val.DATASETS.TEST
    cfg_target1_val.DATASETS.TRAIN = cfg_target1_val.DATASETS.TEST
    cfg_target2_val.DATASETS.TRAIN = cfg_target2_val.DATASETS.TEST
    cfg_source_val.INPUT.MIN_SIZE_TEST = 0
    cfg_target1_val.INPUT.MIN_SIZE_TEST = 0
    cfg_target2_val.INPUT.MIN_SIZE_TEST = 0
    data_loader_source_test = build_detection_train_loader( cfg_source_val)
    data_loader_target1_test = build_detection_train_loader( cfg_target1_val)
    data_loader_target2_test = build_detection_train_loader(cfg_target2_val)
             

    BATCH_SIZE = 4
    data_loader_source_val = build_detection_test_loader(cfg_source_val, cfg_source_val.DATASETS.TEST[0] )
    data_loader_target1_val = build_detection_test_loader(cfg_target1_val, cfg_target1_val.DATASETS.TEST[0])
    data_loader_target2_val = build_detection_test_loader(cfg_target2_val, cfg_target2_val.DATASETS.TEST[0])
 
    source_batch = fill_batch(data_loader_source_val, batch_size=BATCH_SIZE)
    target1_batch = fill_batch(data_loader_target1_val, batch_size=BATCH_SIZE)
    target2_batch = fill_batch(data_loader_target2_val, batch_size=BATCH_SIZE)
    

    batch = {cfg_source_val.DATASETS.TRAIN[0] :source_batch,
            cfg_target1_val.DATASETS.TRAIN[0] :target1_batch,
            cfg_target2_val.DATASETS.TRAIN[0] :target2_batch,
              }
    bdd10k_metadata = MetadataCatalog.get("cityscapes_train_separated" )


    logger.info("Starting training from iteration {}".format(start_iter))

    def aggregate_losses(curr_dict, dest_dict, alpha_dict):
        for da_module_name, _ in alpha_dict.items():
            da_loss_name = f'loss_{da_module_name}'
            if da_module_name in train_alpha_dict and np.isclose(train_alpha_dict[da_module_name], 0.).item():
                continue 
            dest_dict[da_loss_name] += curr_dict[da_loss_name]
    def reduce_losses(comm, loss_distributed_dict, train_alpha_dict, prefix='total'):
        loss_dict_reduced = {}
        for k, v in comm.reduce_dict(loss_distributed_dict).items():
            loss_comp_type = k.split('_')[1]
            if loss_comp_type in train_alpha_dict and np.isclose(train_alpha_dict[loss_comp_type], 0.).item():
                continue 

            loss_dict_reduced[f'{prefix}_{k}' if prefix is not None else k  ] = v.item()
        return loss_dict_reduced

    with EventStorage(start_iter) as storage:
        for data_source, data_target,   data_source_test, data_target1_test, data_target2_test, iteration \
                                        in zip(data_loader_source, data_loader_target1, 
                                                      data_loader_source_test,
                                                      data_loader_target1_test,
                                                      data_loader_target2_test,
                                                      range(start_iter, max_iter)):
            
            storage.step()
            
            model.train()
            
            if ( (iteration+1) % data_len) == 0:
                current_epoch += 1
                i = 1


            i += 1


            # scheduling of the alpha coefficients for domain adaptation, based on percentage of completed training iterations 
            p = iteration/max_iter
            Q = 1
            F = 0.5
            PWR = 3.
            alpha = Q / ( 1. + np.exp( - PWR*p)) - F


            for da_module_name, ub_alpha in ub_alpha_dict.items():
                # clip values of the alphas of the domain adaptation 
                train_alpha_dict[da_module_name] = min(alpha, ub_alpha)

                
            
            loss_dict = model(data_source, False,train_alpha_dict, validation=False)
            loss_dict_target = model(data_target, True, train_alpha_dict, validation=False)

            source_prefix = f'train_source_{cfg_source.DATASETS.TRAIN[0].split("_")[0]  }'
            target_prefix = f'train_target_{cfg_target1.DATASETS.TRAIN[0].split("_")[0]  }'
            source_loss_dict_log = reduce_losses(comm, loss_dict, train_alpha_dict, prefix=source_prefix)
            target_loss_dict_log = reduce_losses(comm, loss_dict_target, train_alpha_dict, prefix=target_prefix)

            aggregate_losses(loss_dict_target, loss_dict, train_alpha_dict)
            for da_module_name, da_alpha in train_alpha_dict.items():
                da_loss_name = f'loss_{da_module_name}'
                # loss is already zeroed by the backward, if alpha is set 0, hence the loggers have to also report the correct value, as the multiplication 
                # by da_alpha=0 happens 
                # in the backward call of the gradreverse layer 
                loss_dict[da_loss_name] *= (lambda_  if da_alpha > 0. else 0    )  

                k = f'{source_prefix}_{da_loss_name}'
                if k in source_loss_dict_log:
                    source_loss_dict_log[f'{source_prefix}_{da_loss_name}'] *= (lambda_  if da_alpha > 0. else 0    )  
                k = f'{target_prefix}_{da_loss_name}'
                if k in target_loss_dict_log:
                    target_loss_dict_log[f'{target_prefix}_{da_loss_name}'] *= (lambda_  if da_alpha > 0. else 0    )

            source_losses_reduced_log = sum(loss for loss in source_loss_dict_log.values())
            target_losses_reduced_log = sum(loss for loss in target_loss_dict_log.values())                

            losses = sum(loss_dict.values())
            # it can happen that train diverges due to the interaction between each task loss and the ones from the domain adversarial components,
            # hence, this check is required
            assert torch.isfinite(losses).all(), loss_dict

            total_loss_dict_log = reduce_losses(comm, loss_dict, train_alpha_dict, prefix='total')
            total_losses_reduced_log = sum(loss for loss in total_loss_dict_log.values())
            if comm.is_main_process():
                storage.put_scalars(train_total_loss=total_losses_reduced_log, **total_loss_dict_log)
                storage.put_scalars(train_source_total_loss=source_losses_reduced_log, **source_loss_dict_log)
                storage.put_scalars(train_target_total_loss=target_losses_reduced_log, **target_loss_dict_log)


                alphas_dict = { f'alpha_{k}':v for k,v in train_alpha_dict.items()}
                storage.put_scalars(**alphas_dict)         
                
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            



            if (
                cfg_source.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg_source.TEST.EVAL_PERIOD == 0

            ):




             
                model.eval()

                val_source_prefix = f'val_source_{cfg_source.DATASETS.TRAIN[0].split("_")[0]  }'
                val_target1_prefix = f'val_target1_{cfg_target1.DATASETS.TRAIN[0].split("_")[0]  }'
                val_target2_prefix = f'val_target2_{cfg_target2.DATASETS.TRAIN[0].split("_")[0]  }'


                res_target1 = do_test(cfg_target1, model)
                res_target2 = do_test(cfg_target2, model)
                for k, metrics in res_target1.items():
                    res_target1[k] = { f'{val_target1_prefix}_{k}': v   for k,v in metrics.items() }
                for k, metrics in res_target2.items():                    
                    res_target2[k] = { f'{val_target2_prefix}_{k}': v   for k,v in metrics.items() }

                # avoid storing the gradients when computing the validation loss
                with torch.no_grad():

                    model.train()
                    source_preds_loss = model(data_source_test, False,train_alpha_dict, validation=True)
                    target1_preds_loss = model(data_target1_test, True,train_alpha_dict, validation=True)
                    target2_preds_loss = model(data_target2_test, True,train_alpha_dict, validation=True)

                    
                    source_loss_dict_log = reduce_losses(comm, source_preds_loss, train_alpha_dict, prefix=val_source_prefix)
                    target1_loss_dict_log = reduce_losses(comm, target1_preds_loss, train_alpha_dict, prefix=val_target1_prefix)
                    target2_loss_dict_log = reduce_losses(comm, target2_preds_loss, train_alpha_dict, prefix=val_target2_prefix )

                    val_losses_dict_log = reduce_losses(comm, source_preds_loss, train_alpha_dict, prefix=None)#copy.deepcopy(source_loss_dict_log)

                    aggregate_losses(reduce_losses(comm, target1_preds_loss, train_alpha_dict, prefix=None), val_losses_dict_log, train_alpha_dict)
                    aggregate_losses(reduce_losses(comm, target2_preds_loss, train_alpha_dict, prefix=None ), val_losses_dict_log, train_alpha_dict)
                    
                    for da_module_name, da_alpha in train_alpha_dict.items():
                        da_loss_name = f'loss_{da_module_name}'

                        if da_loss_name in val_losses_dict_log: 
                            val_losses_dict_log[da_loss_name] *= (lambda_  if da_alpha > 0. else 0    )    


                        k =f'{val_source_prefix}_{da_loss_name}'
                        if k in source_loss_dict_log:
                            source_loss_dict_log[k] *= (lambda_  if da_alpha > 0. else 0    )  

                        k = f'{val_target1_prefix}_{da_loss_name}'
                        if k in target1_loss_dict_log:
                            target1_loss_dict_log[k] *= (lambda_  if da_alpha > 0. else 0    )  

                        k = f'{val_target2_prefix}_{da_loss_name}'
                        if k in target2_loss_dict_log:
                            target2_loss_dict_log[k] *= (lambda_  if da_alpha > 0. else 0    )  

                    source_losses_reduced_log = sum(loss for loss in source_loss_dict_log.values())
                    target1_losses_reduced_log = sum(loss for loss in target1_loss_dict_log.values())
                    target2_losses_reduced_log = sum(loss for loss in target2_loss_dict_log.values())                    
                            
                    val_losses_reduced = sum(val_losses_dict_log.values())

                    if comm.is_main_process():
                        val_losses_dict_log = {f'val_total_{k}':v for k,v in val_losses_dict_log.items()  }
                        storage.put_scalars(val_total_loss=val_losses_reduced, **val_losses_dict_log)  

                        storage.put_scalars(val_source_total_loss=source_losses_reduced_log, **source_loss_dict_log)  
                        storage.put_scalars(val_target1_total_loss=target1_losses_reduced_log, **target1_loss_dict_log) 
                        storage.put_scalars(val_target2_total_loss=target2_losses_reduced_log, **target2_loss_dict_log)  

                        for k, metrics in res_target1.items():
                            storage.put_scalars(**metrics)  
                        for k, metrics in res_target2.items():
                            storage.put_scalars(**metrics)  
                        model.eval()
                        
                        for batch_name, curr_batch in batch.items():
                            outputs = get_pred(model, curr_batch, bdd10k_metadata)
                            for idx, output in enumerate(outputs):
                                torch_uint8_image = torch.ByteTensor(output).permute(2,0,1)
                                storage.put_image(f'{batch_name} {idx}', torch_uint8_image)

                        # the three line of code below are an alternative way to the immediate above for loop, to store prediction results in a rectangular grid
                        # of images

                        #tensor = torch.Tensor(outputs).permute((0, 3,1,2))
                        #grid_results = torchvision.utils.make_grid(tensor, nrow=4)
                        #storage.put_image('image predictions', grid_results)


                    #grid_numpy = grid_results.numpy().astype(np.uint8).transpose(1,2,0)
                    
                    
                
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()


            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)





def setup(args):
    """
    Create configs and perform basic setups.
    """
    
    

    cfg = get_cfg()
    
    add_da_panfpn_config(cfg)


    cfg.DATASETS.TRAIN = (cfg.DA.TRAIN.SOURCE_DATASET_NAME, )
    cfg.DATASETS.TEST = ( cfg.DA.TEST.SOURCE_DATASET_NAME,)
    cfg.SOLVER.IMS_PER_BATCH = cfg.DA.TRAIN.SOURCE_BATCH_SIZE 
    cfg.OUTPUT_DIR = os.path.join(cfg.LOG.LOG_FOLDER, cfg.DATASETS.TRAIN[0])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    cfg_target1 = copy.deepcopy(cfg)
    cfg_target1.DATASETS.TRAIN = (cfg.DA.TRAIN.TARGET_DATASET_NAME,)
    cfg_target1.DATASETS.TEST = (cfg.DA.TEST.TARGET_DATASET_NAME,)
    cfg_target1.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_target1.SOLVER.IMS_PER_BATCH = cfg.DA.TRAIN.TARGET_BATCH_SIZE
    cfg_target1.OUTPUT_DIR = os.path.join(cfg.LOG.LOG_FOLDER, cfg_target1.DATASETS.TRAIN[0])
    os.makedirs(cfg_target1.OUTPUT_DIR, exist_ok=True)

    cfg_target2 = copy.deepcopy(cfg)
    cfg_target2.DATASETS.TRAIN = (cfg.DA.TRAIN.OUT_OF_SAMPLE_DATASET_NAME ,)
    cfg_target2.DATASETS.TEST = (cfg.DA.TEST.OUT_OF_SAMPLE_DATASET_NAME,)
    cfg_target2.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_target2.SOLVER.IMS_PER_BATCH = cfg.DA.TRAIN.TARGET_BATCH_SIZE
    cfg_target2.OUTPUT_DIR = os.path.join(cfg.LOG.LOG_FOLDER, cfg_target2.DATASETS.TRAIN[0])
    os.makedirs(cfg_target2.OUTPUT_DIR, exist_ok=True)


    default_setup(
        cfg, args
    )  
    default_setup(
        cfg_target1, args
    )  
    default_setup(
        cfg_target2, args
    )  


    return cfg,cfg_target1, cfg_target2


def main(args):
    cfg_source, cfg_target1, cfg_target2 = setup(args)
    cfg_target = cfg_target1
    model = build_model(cfg_source)

    register_datasets(args)




    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg_source.OUTPUT_DIR).resume_or_load(
            cfg_source.MODEL.WEIGHTS, resume=cfg_source.resume
        )
        return do_test(cfg_target, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg_source, cfg_target1, cfg_target2,  model, resume=args.resume)
    res_source = do_test(cfg_source, model)
    res_target1 = do_test(cfg_target1, model)
    res_target2 = do_test(cfg_target2, model)
    
    out_path_source = os.path.join(cfg_source.LOG.LOG_FOLDER, cfg_source.DATASETS.TEST[0] + '_test_results.json')
    out_path_target1 = os.path.join(cfg_target1.LOG.LOG_FOLDER, cfg_target1.DATASETS.TEST[0] + '_test_results.json')
    out_path_target2 = os.path.join(cfg_target2.LOG.LOG_FOLDER, cfg_target2.DATASETS.TEST[0] + '_test_results.json')
    save_json(res_source, out_path_source)
    save_json(res_target1, out_path_target1)
    save_json(res_target2, out_path_target2)

    return do_test(cfg_target2, model)



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--data-root", help="path to the top directory containing all datasets")

    args = parser.parse_args()

    print( default_argument_parser())
    print(type( default_argument_parser()))
    print(type(args))
    launch(
            main,
            args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        )
