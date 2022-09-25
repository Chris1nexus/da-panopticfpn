# INSTANCES ONLY
python3 carla_to_coco_det.py --datadir $DATA_ROOT/carla_datasets \
--image-dirpath $DATA_ROOT/coco_carla/train_images \
--annot-dirpath $DATA_ROOT/coco_carla/train_annotations \
--output-json-file instances_things.json --things-only --use-train-id
#STUFF_ONLY
python3 carla_to_coco_det.py --datadir $DATA_ROOT/carla_datasets \
--image-dirpath $DATA_ROOT/coco_carla/train_images \
--annot-dirpath $DATA_ROOT/coco_carla/train_annotations \
--output-json-file stuff.json --stuff-only --use-train-id
#PANOPTIC_ANNOTATIONS (REQUIRED BY PANOPTIC_TO_PANFPN.PY)
python3 carla_to_coco_det.py --datadir $DATA_ROOT/carla_datasets \
--image-dirpath $DATA_ROOT/coco_carla/train_images \
--annot-dirpath $DATA_ROOT/coco_carla/train_annotations \
--output-json-file instances_all.json --panoptic-only  --use-train-id
#2. SECOND STEP
#CONVERT COCO DET TO COCO PANOPTIC
python3 coco_det_to_coco_pan.py --input_json_file $DATA_ROOT/coco_carla/train_annotations/instances_all.json \
--output_json_file $DATA_ROOT/coco_carla/train_annotations/panoptic.json \
--segmentations_folder $DATA_ROOT/coco_carla/train_panoptic  --nproc 2 --use-train-id
#3. THIRD STEP
#CONVERT COCO PANOPTIC TO COCO PANFPN
python3 panoptic_to_pan_fpn.py  --panoptic_json_file $DATA_ROOT/coco_carla/train_annotations/panoptic.json  --panoptic_root $DATA_ROOT/coco_carla/train_panoptic  --sem_seg_root $DATA_ROOT/coco_carla/train_sem_stuff    --categories_stuff_panfpn_json $DATA_ROOT/coco_carla/categories_stuff_panfpn.json --nproc 10