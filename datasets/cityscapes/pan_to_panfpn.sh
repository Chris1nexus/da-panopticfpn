python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_train_trainId.json \
--panoptic_root $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_train_trainId  \
--sem_seg_root $DATA_ROOT/pan_cityscapes_reduced/train_sem_stuff  \
--categories_stuff_panfpn_json $DATA_ROOT/pan_cityscapes_reduced/categories_stuff_panfpn.json \
--nproc 10
python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_val_trainId.json \
--panoptic_root $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_val_trainId  \
--sem_seg_root $DATA_ROOT/pan_cityscapes_reduced/val_sem_stuff  \
--categories_stuff_panfpn_json $DATA_ROOT/pan_cityscapes_reduced/categories_stuff_panfpn.json \
--nproc 10
python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_test_trainId.json \
--panoptic_root $DATA_ROOT/pan_cityscapes_reduced/cityscapes_panoptic_test_trainId  \
--sem_seg_root $DATA_ROOT/pan_cityscapes_reduced/test_sem_stuff  \
--categories_stuff_panfpn_json $DATA_ROOT/pan_cityscapes_reduced/categories_stuff_panfpn.json \
--nproc 10