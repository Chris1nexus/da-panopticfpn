# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict, List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList

from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head







import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, sigmoid_focal_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage


from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
__all__ = ["DAPanopticFPN"]


from .fpn import FPN
from .da import (DiscriminatorRes2,
            DiscriminatorRes3,
            DiscriminatorRes4,
            DiscriminatorRes5,
            DiscriminatorP,
            )




@META_ARCH_REGISTRY.register()
class DAPanopticFPN(GeneralizedRCNN):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    resnet_da_template = {'res2': DiscriminatorRes2,
        'res3': DiscriminatorRes3,
        'res4': DiscriminatorRes4,
        'res5': DiscriminatorRes5}
    
    fpn_da_template = {'p2': DiscriminatorP,
        'p3': DiscriminatorP,
        'p4': DiscriminatorP,
        'p5': DiscriminatorP,
        'p6': DiscriminatorP
                    }
    
    alpha_template = {
        'res2': 0.5,
        'res3': 0.5,
        'res4': 0.5,
        'res5': 0.5,
        'p2': 0.5,
        'p3': 0.5,
        'p4': 0.5,
        'p5': 0.5,
        'p6': 0.5,
    }
    @configurable
    def __init__(
        self,
        *, # python Bare functionality to enforce its following args to be 'keyword args' 
        sem_seg_head: nn.Module, # backbone and other stuff is added to the members of this instance by the superclass
                    # custom modules and other args are either passed with their keyword names, if specified
                    # or grouped in the last **kwargs if not (these are all keyword args contained in the cfg file)
                    # returned by the class method from_config. Hence, everything that is created in from_config of this
                    # current class, and added to the dictionary with its keyword, is found again here, either as specified keyword
                    # like sem_seg_head, or as dictionary entry, in **kwargs
        combine_overlap_thresh: float = 0.5,
        combine_stuff_area_thresh: float = 4096,
        combine_instances_score_thresh: float = 0.5,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold
        Other arguments are the same as :class:`GeneralizedRCNN`.
        """
        super().__init__(**kwargs)
        
        self.sem_seg_head = sem_seg_head
        # options when combining instance & semantic outputs
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh
        
        
        
        fpn_out_shape = self.backbone.output_shape()
        resnet_out_shape = self.backbone.bottom_up.output_shape()
        
        self.resnet_da_dict = dict()
        self.fpn_da_dict = dict()
        
        for feature_name, feature_info in fpn_out_shape.items():
            n_channels = feature_info.channels
            da_model = DAPanopticFPN.fpn_da_template[feature_name](n_channels)
            self.fpn_da_dict[feature_name] = da_model
            self.add_module(feature_name, da_model)
            
        for feature_name in resnet_out_shape.keys():
            da_model = DAPanopticFPN.resnet_da_template[feature_name]()
            self.resnet_da_dict[feature_name] = da_model
            self.add_module(feature_name, da_model)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        # build_sem_seg_head is not called in build generalizedRCNN, hence it requires to be instantiated 
        # manually
        ret.update(
            {
                "combine_overlap_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH,
                "combine_stuff_area_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT,
                "combine_instances_score_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH,  # noqa
            }
        )
        ret["sem_seg_head"] = build_sem_seg_head(cfg, ret["backbone"].output_shape())
        logger = logging.getLogger(__name__)
        if not cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED:
            print('comb_not_enabled')
            logger.warning(
                "PANOPTIC_FPN.COMBINED.ENABLED is no longer used. "
                " model.inference(do_postprocess=) should be used to toggle postprocessing."
            )
        if cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT != 1.0:
            print('inst_wt_loss != 1.0')
            w = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT
            logger.warning(
                "PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT should be replaced by weights on each ROI head."
            )

            def update_weight(x):
                if isinstance(x, dict):
                    return {k: v * w for k, v in x.items()}
                else:
                    return x * w

            roi_heads = ret["roi_heads"]
            roi_heads.box_predictor.loss_weight = update_weight(roi_heads.box_predictor.loss_weight)
            roi_heads.mask_head.loss_weight = update_weight(roi_heads.mask_head.loss_weight)

        return ret

    def forward(self, batched_inputs,  target_domain = False, alpha_dict=alpha_template, validation=False):#alpha3 = 1, alpha4 = 1, alpha5 = 1):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)


        fpn_features, resnet_features = self.backbone(images.tensor)
        

        losses = dict()
        
        # domain adaptation
        #### fpn DA
        for feat_name, feat_tensor in fpn_features.items():
            da_module = self.fpn_da_dict[feat_name]
            loss = da_module(fpn_features[feat_name], 
                      target_domain=target_domain,
                      alpha=alpha_dict[feat_name])
            losses[f'loss_{feat_name}'] = loss
            
        #### resnet DA
        for feat_name, feat_tensor in resnet_features.items():
            da_module = self.resnet_da_dict[feat_name]
            loss = da_module(resnet_features[feat_name], 
                      target_domain=target_domain,
                      alpha=alpha_dict[feat_name])
            losses[f'loss_{feat_name}'] = loss
        
        if target_domain and not validation:
            return losses
          

        # semantic segmentation
        assert "sem_seg" in batched_inputs[0]
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
        ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(fpn_features, gt_sem_seg)
        
        # instance segmentation
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(images, fpn_features, gt_instances)
        detector_results, detector_losses = self.roi_heads(
            images, fpn_features, proposals, gt_instances
        )

        losses.update(sem_seg_losses)
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
        images = self.preprocess_image(batched_inputs)
        fpn_features, resnet_features = self.backbone(images.tensor)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(fpn_features, None)
        proposals, _ = self.proposal_generator(images, fpn_features, None)
        detector_results, _ = self.roi_heads(images, fpn_features, proposals, None)

        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.
    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id
    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info