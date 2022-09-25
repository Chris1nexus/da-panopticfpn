python3 -m to_coco_panseg_reduced -i $DATA_ROOT/bdd/bdd100k_pan_seg_labels_trainval/bdd100k/labels/pan_seg/bitmasks/train \
--nproc 32 -pb $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced/train \
-o $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced_train.json
python3 -m to_coco_panseg_reduced -i $DATA_ROOT/bdd/bdd100k_pan_seg_labels_trainval/bdd100k/labels/pan_seg/bitmasks/val \
--nproc 32 -pb $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced/val \
-o $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced_val.json