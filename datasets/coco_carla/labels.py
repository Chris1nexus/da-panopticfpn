from collections import namedtuple
import json
from panopticapi.utils import json
import argparse
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )
labels = [
    #name id trainId category catId hasInstances   ignoreInEval   color
Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
Label("building", 1, 2, "construction", 2, False, False, (70, 70, 70)),
Label("fence", 2, 4, "construction", 2, False, False, (190, 153, 153)),
Label("other", 3, 255, "void", 0, False, True, (55, 90, 80)),
Label("person", 4, 11, "human", 6, True, False, (220, 20, 60)),
#Label(  'rider', 25 ,12 , 'human' , 6, True, False , (255,  0,  0) ),
Label("pole", 5, 5, "object", 3, False, False, (153, 153, 153)), 

# as the reverse convention is used for trainId2label, road should come first, as it encompasses roadlines
Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),       
Label("roadline", 6, 0, "flat", 1, False, False, (128, 64, 128)),
Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),  
Label("vegetation", 9, 8, "nature", 4, False, False, (107, 142, 35)),
Label("car", 10, 13, "vehicle", 7, True, False, (0, 0, 142)),
#Label("vehicle", 10, 13, "vehicle", 7, True, False, (0, 0, 142)),
Label("wall", 11, 3, "construction", 2, False, False, (102, 102, 156)),
Label("traffic sign", 12, 7, "object", 3, False, False, (220, 220, 0)),
Label("sky", 13, 10, "sky", 5, False, False, (70, 130, 180)),
Label("ground", 14, 255, "void", 0, False, True, (81, 0, 81)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("rail track", 16, 255, "flat", 1, False, True, (230, 150, 140)),
Label(
        "guard rail", 17, 255, "construction", 2, False, True, (180, 165, 180)
    ),
Label("traffic light", 18, 6, "object", 3, False, False, (250, 170, 30)),
Label("static", 19, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 20, 255, "void", 0, False, True, (111, 74, 0)),
    Label("water", 21, 255, "void", 0, False, True, (110, 190, 160)),
Label("terrain", 22, 9, "nature", 4, False, False, (152,251,152)),

]


name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]



if __name__ =='__main__':


    parser = argparse.ArgumentParser(
        description="This script creates the categories.json needed for COCO format conversions. \
            It follows the cityscapes convention."
    )
    parser.add_argument('--categories-json-path', type=str,
                        help="file path to which the JSON containing cityscapes like categories will be saved",
                        default='./categories_carla_cityscapes.json')
    args = parser.parse_args()
    import json
    categories = []
    for item in labels:
        item_dict = item._asdict()
        categories.append(item_dict)

    with open(args.categories_json_path, 'w') as f:  
        json.dump(categories, f)