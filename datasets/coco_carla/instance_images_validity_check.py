import json
import cv2
from tqdm import tqdm
import argparse


def median_filtering(inst_img, ker_size=(9,9)):
    """

        inst_img in RGB format (containing semantic id in R channel)

        credits to fmw42 's post https://stackoverflow.com/questions/61565277/python-image-processing-how-to-remove-certain-contour-and-blend-the-value-with


    """
    # anomalies are masked as white
    R = np.where(inst_img[...,0] > 22, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ker_size )
    
    # dilate anomalous pixels so that they cover a wider area: necessary to then find the neighborhood with which the anomalous pixels
    # are replaced
    mask = cv2.morphologyEx(R, cv2.MORPH_DILATE, kernel)
    # make mask 3 channel
    mask = cv2.merge([mask,mask,mask])
    # invert mask
    mask_inv = 255 - mask
    # get area of largest contour of the white masked dilated regions
    contours = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    perimeter_max = 0
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > perimeter_max:
            perimeter_max = perimeter

    # approx radius from largest area
    radius = int(perimeter_max/2) + 1
    if radius % 2 == 0:
        radius = radius + 1

    # median filter input image so that all areas with outliers are replaced by median values (those of the instance mask)
    median = cv2.medianBlur(inst_img, radius)

    # apply mask to image: non anomalous pixels are kept
    img_masked = cv2.bitwise_and(inst_img, mask_inv)

    # apply inverse mask to median: areas that are cleaned from anomalous pixels are kept
    median_masked = cv2.bitwise_and(median, mask)

    # add together: cleaned areas fill the empty spaces that previously contained anomalous pixels
    result = cv2.add(img_masked,median_masked)

    return result


def validate_istance_images(instance_coco_json_file, categories_json_file, original_datasets_dirpath, overwrite, log_filepath):

    with open(instance_coco_json_file) as f:
        carla_dict = json.load(f)
    with open(categories_json_file) as f:
        categories_list = json.load(f)
        
    categories = {category['id']: category for category in categories_list} 



    cat_set = set()
    cat_wrong_id = []

    for ann in carla_dict['annotations']:
        
        if ann['category_id'] not in categories:
            cat_wrong_id.append(ann)
        cat_set.add(ann['category_id'])
    image_annot_dict = dict()
    wrong_images_ids = set([cat_dict['image_id']  for cat_dict in  cat_wrong_id ])
    image_to_annots = dict()
    for ann in carla_dict['annotations']:
        if ann['image_id'] in wrong_images_ids:
            if ann['image_id']  not in image_to_annots:
                image_to_annots[ann['image_id']] = []
            image_to_annots[ann['image_id']].append(ann)
    wrong_images = dict()
    for image_dict in carla_dict['images']:
        if image_dict['id'] in wrong_images_ids:
            wrong_images[image_dict['id']] = image_dict



    datasets_dirpath = original_datasets_dirpath

    artifacted = []

    progressbar = tqdm(total=len(wrong_images.items()) )
    for image_id, wrong_image_dict in wrong_images.items():
        wrong_img_filename = wrong_image_dict['file_name'].replace('rgb', 'instance_segmentation')
        filepath = wrong_img_filename.replace('--', '/')
        
        
        orig_filepath = os.path.join(datasets_dirpath, filepath)
        
        inst_img = cv2.imread(orig_filepath)[:,:,::-1]
        
        
        mod_diff_len = 1
        ker_size =(5,5)
        
        outlier_px_query = inst_img[...,0] > 22
        ids_to_remove_set = set(rgb2id(inst_img)[outlier_px_query])
        
        result_inst_img =  median_filtering(inst_img, ker_size=ker_size )

        intersec_len, orig_diff_len, mod_diff_len = set_stats(inst_img, result_inst_img)
        
        new_ids_set = set(rgb2id(result_inst_img)[outlier_px_query])
        
        disparity_len = len(ids_to_remove_set.intersection(new_ids_set))
        old_id_set = set(np.unique(rgb2id(inst_img)))
        new_id_set = set(np.unique(rgb2id(result_inst_img)))
        num_artifacted_ids = len(new_id_set - old_id_set)
        has_outlier_categories = (result_inst_img[...,0] > 22).any()
        
        if num_artifacted_ids > 0:
            print(orig_filepath, " ", num_artifacted_ids)
            #artifacted.append({'img': inst_img, 'res':result_inst_img, 'n_artifacts':num_artifacted_ids}  )
            curr_ids_list = list(ids_to_remove_set)
            artifacted.append({'ref_path': orig_filepath,
                'num_artifacted_pixels': outlier_px_query.sum() ,
                'artifacted_ids':[curr_ids_list]
                })
        
        
        assert disparity_len == 0 and  has_outlier_categories == False, f"Error: {disparity_len},{num_artifacted_ids}, {has_outlier_categories} "
        if overwrite:
            cv2.imwrite(orig_filepath, result_inst_img [:,:,::-1])
        progressbar.update(1)
    with open(log_filepath, 'w') as f:
        json.dump(artifacted, f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script converts panoptic COCO format to  the format required by the panoptic FPN. See this file's head for more information."
    )
    parser.add_argument('--instance_coco_json_file', type=str,
                        help="JSON file with panoptic COCO format annotations")
    parser.add_argument('--original_datasets_dirpath', type=str,
                        help="Folder that contains carla generated datasets")
    parser.add_argument("--overwrite", type=str2bool, nargs='?',
                        const=True, default=False,
                            help="(default: False) Overwrite images which contain artifacts, with their postprocessed version."+\
                            "The path to artifacted images is stored in both cases in ./artifact_details.json")

    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--log_filepath', type=str,
                        help="JSON file containing the paths of artifacted images",
                        default='./artifact_details.json')
    args = parser.parse_args()

    validate_istance_images(args.instance_coco_json_file, args.categories_json_file, args.original_datasets_dirpath, args.overwrite, args.log_filepath)