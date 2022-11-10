# da-panopticfpn

A **sim2real panoptic segmentation model** for driving scene understanding.

To achieve this objective, a panoptic segmentation dataset composed of synthetic and real driving scene images has been developed.

The former has been obtained by means of simulation on the **CARLA simulator**.

The real world components have been integrated by means of the renowned [**Cityscapes**](https://www.cityscapes-dataset.com/downloads/) and [**BDD100k dataset**](https://www.bdd100k.com/).

To merge the three into a common sim2real dataset, a common subset of labels relevant for driving scene understanding has been determined and utilized
to re-map the ground-truths of each dataset to a common format.

The sim2real setting proposed in this work considers the training of a panoptic segmentation model that exploits only synthetic data to learn the way to accomplish its main task, while the real world data is only exploited by a self-supervised task.

Nevertheless, this multi-task setting allows the model to transfer its panoptic segmentation capability across the domains,
improving upon models solely trained on synthetic data and tested on real driving scenes. 

As such, only synthetic data annotations have been used to train the model, while real data has been employed only as a means to accomplish a self supervised domain classification task and to evaluate model performance on the task.

Finally, the model architecture is optimized in order to develop a configuration that maximizes the transferred panoptic segmentation performance from synthetic to the real world. Further details are provided at the link of the publication https://webthesis.biblio.polito.it/22581/1/tesi.pdf


## Dataset setup
All three datasets have to be stored in a common DATA_ROOT folder, which can be configured by modifying the script **set_env.sh** with the appropriate path.
Then, to set the environment variable for the current shell, simply run:
```bash
source set_env.sh
```
The DATA_ROOT folder must contain three main folder named:
- cityscapes
- bdd
- coco_carla

#### Cityscapes
Download the following cityscapes subsets from the official (cityscapes website)[https://www.cityscapes-dataset.com/downloads/]:
- gtFine_trainvaltest.zip
- leftImg8bit_trainvaltest.zip
Each must be put within the **cityscapes** folder as described above and unzipped there.

#### BDD100K
Download the following bdd10k subsets from the official (BDD100K website)[https://bdd-data.berkeley.edu/portal.html#download]:
- panoptic segmentation annotations
- bdd10k images (note that it is the 10k subset, not the 100k one)

Each must be put within the **bdd** folder as described above and unzipped there.

#### COCO Carla
Download from google cloud bucket. Link soon to be available.


## Sim2Real Dataset creation
To this end, the (COCO format)[https://cocodataset.org/#format-results] serves as the base protocol to gather and store data uniformly across each subset.
Hence, each dataset has been mapped to a common format following the same data pipeline.

The common steps are as follows:
1. **(dataset-specific labels -> common label set)** map each dataset-specific category label id to the common category label set id
2. **(common label set -> COCO-detection)** map instance and semantic segmentation masks to the COCO-detection format (for instance masks)
3. **(COCO-detection -> COCO-panoptic)** instance masks and semantic segmentation masks are mapped to panoptic segmentation masks according to the COCO-panoptic format
4. **(COCO-panoptic -> Detectron2 panopticFPN)** COCO-Panoptic masks are mapped to the Panoptic Segmentation format required by the (PanopticFPN model)[https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts] (read the note about the panopticfpn architecture at the linked page )



## Training
The **training** and **hyper-parameter** optimization can both be carried out from the provided jupyter notebook.
It provides a simple interface to track experiments that have already been executed as well as new ones designed by changing the list of parameters of the grid-search dictionary.

## Visualization of the metrics and qualitative panoptic predictions
After training or hyper-parameter optimization has been carried out, all metrics can be visualized by running from the top level folder of this project:
```bash
tensorboard --logdir .
```

