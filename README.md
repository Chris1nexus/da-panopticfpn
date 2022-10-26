# da-panopticfpn

A sim2real panoptic segmentation model for driving scene understanding.

To achieve this objective, a panoptic segmentation dataset composed of synthetic and real driving scene images has been developed.
The former has been obtained by means of simulation on the CARLA simulator.
The real world components have been integrated by means of the renowned (Cityscapes)[https://www.cityscapes-dataset.com/downloads/] and (BDD100k dataset)[https://www.bdd100k.com/].
To merge the three into a common sim2real dataset, a common subset of labels relevant for driving scene understanding has been determined and utilized
to re-map the ground-truths of each dataset to a common format.

To this end, the (COCO format)[https://cocodataset.org/#format-results] serves as the base protocol to gather and store data uniformly across each subset.
Hence, each dataset has been mapped to a common format following the same data pipeline.
The common steps are as follows:
1. map each dataset-specific category label id to the common category label set id
2. map instance and semantic segmentation masks to the COCO-detection format (for instance masks)
3. instance masks and semantic segmentation masks are mapped to panoptic segmentation masks according to the COCO-panoptic format
4. COCO-Panoptic masks are mapped to the Panoptic Segmentation format required by the (PanopticFPN model)[https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts] (read the note about the panopticfpn architecture at the linked page )



## Setup 



## Dataset


## Training

