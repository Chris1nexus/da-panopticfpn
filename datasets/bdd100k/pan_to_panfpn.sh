python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced_train.json \
--panoptic_root $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced/train \
--sem_seg_root $DATA_ROOT/bdd100k_reduced/train_sem_stuff  \
--categories_stuff_panfpn_json $DATA_ROOT/bdd100k_reduced/categories_stuff_panfpn.json \
--nproc 10
python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced_val.json \
--panoptic_root $DATA_ROOT/bdd100k_reduced/bdd100k_panoptic_reduced/val \
--sem_seg_root $DATA_ROOT/bdd100k_reduced/val_sem_stuff  \
--categories_stuff_panfpn_json $DATA_ROOT/bdd100k_reduced/categories_stuff_panfpn.json \
--nproc 10