python3 carla_to_coco_det.py --datadir $DATA_ROOT/carla_datasets_test --image-dirpath $DATA_ROOT/coco_carla/val_images --annot-dirpath $DATA_ROOT/coco_carla/val_annotations --output-json-file instances_things.json --things-only --use-train-id
python3 carla_to_coco_det.py --datadir $DATA_ROOT/carla_datasets_test --image-dirpath $DATA_ROOT/coco_carla/val_images --annot-dirpath $DATA_ROOT/coco_carla/val_annotations --output-json-file instances_all.json --panoptic-only  --use-train-id
python3 coco_det_to_coco_pan.py --input_json_file $DATA_ROOT/coco_carla/val_annotations/instances_all.json \
--output_json_file $DATA_ROOT/coco_carla/val_annotations/panoptic.json \
--segmentations_folder $DATA_ROOT/datasets/coco_carla/val_panoptic  --nproc 8 --use-train-id
python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/coco_carla/val_annotations/panoptic.json \
--panoptic_root $DATA_ROOT/coco_carla/val_panoptic --sem_seg_root $DATA_ROOT/coco_carla/val_sem_stuff \
--categories_stuff_panfpn_json $DATA_ROOT/coco_carla/categories_stuff_panfpn.json --nproc 32