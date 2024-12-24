import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PersonalizeSAM.per_segment_anything import SamPredictor, sam_model_registry
from algorithms import PerSamPredictor, run_per_segment_anything, run_p2sam, load_model
from constants import *
from P2SAM.eval_utils import AverageMeter, intersectionAndUnion
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from model_api.models.result_types.visual_prompting import ZSLVisualPromptingResult
from model_api.models.visual_prompting import Prompt, SAMLearnableVisualPrompter
from utils import get_colors, load_dataset, save_visualization

import datumaro as dm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam_name', type=str, default="SAM", choices=MODEL_MAP.keys())
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--algo',type=str, default='persam', choices=['persam', 'p2sam'])
    parser.add_argument("--num_priors", type=int, default=1, help="Number of prior images to use as references")
    parser.add_argument("--dataset_name", type=str, default="PerSeg", choices=DATASETS)
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--show", action="store_true", help="Show results during processing")
    parser.add_argument("--post_refinement", action="store_true", help="Apply post refinement")
    parser.add_argument("--class_name", type=str, default=None)

    args = parser.parse_args()
    return args

def predict_on_dataset(args: argparse.Namespace, predictor: PerSamPredictor, dataframe: pd.DataFrame, model_name: str, algo_name: str) -> tuple[pd.DataFrame, float, float]:
    result_dataframe = pd.DataFrame(columns=['class_name', 'IoU', 'Accuracy'])

    if args.class_name:  # filter on class_name
        dataframe = dataframe[dataframe.class_name == args.class_name]  

    for class_name in tqdm(dataframe.class_name.unique(), desc="Processing classes", total=len(dataframe.class_name.unique()), position=0, leave=True):
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        class_samples = dataframe[dataframe.class_name == class_name]
        priors = class_samples.head(args.num_priors)
        # select remaining images as target images but do not change the order of the dataframe 
        targets = class_samples[~class_samples.index.isin(priors.index)]

        # learn on prior images
        if isinstance(predictor, SAMLearnableVisualPrompter):
            raise NotImplementedError("MAPI implementation does not accept masks directly. Either convert to polygons or change the implementation")    
        
        # Load all prior images and masks
        prior_images = []
        prior_masks = []
        for _, prior in priors.iterrows():
            # load image from disk and convert to numpy array
            image = cv2.cvtColor(cv2.imread(prior.image), cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(cv2.imread(prior.mask_image), cv2.COLOR_BGR2RGB)
            mask_prompt = Prompt(label=0, data=mask_image)  # TODO only single class per image is supported for now
            prior_images.append(image)
            prior_masks.append(mask_prompt)

        if args.save:
            # save prior images and masks to disk, on top of each other
            for i, (image, mask) in enumerate(zip(prior_images, prior_masks)):
                cv2.imwrite(f"outputs/{args.dataset_name}/{class_name}/prior_{i}.png", np.hstack([image, mask]))

        # TODO currently multiple priors is not yet implemented
        predictor.learn(image=prior_images[0], masks=prior_masks, show=args.show)

        # predict on target images
        for row_idx, target in tqdm(targets.iterrows(), desc="Processing samples", total=len(targets), position=1, leave=False):
            # load image from disk and convert to numpy array
            target_image = cv2.cvtColor(cv2.imread(target.image), cv2.COLOR_BGR2RGB)
            gt_mask = cv2.cvtColor(cv2.imread(target.mask_image), cv2.COLOR_BGR2RGB)
            
            result: ZSLVisualPromptingResult = predictor.infer(target_image=target_image, apply_masks_refinement=args.post_refinement)

            mask = result.get_mask(0)
            # Merge all instance masks into one mask using logical OR
            merged_mask = np.zeros_like(mask.mask[0], dtype=bool)
            for instance in mask.mask:
                merged_mask = np.logical_or(merged_mask, instance)

            if args.save:
                output_path = os.path.join('outputs', args.dataset_name, class_name, os.path.basename(target.image))
                save_visualization(
                    image=target_image,
                    mask=mask,
                    output_path=output_path,
                    points=mask.points if hasattr(mask, 'points') else None,
                    scores=mask.scores if hasattr(mask, 'scores') else None
                )
            
            # Convert to uint8 for comparison with gt
            mask = np.uint8(merged_mask)
            gt_mask = np.uint8(gt_mask > 0)
            
            intersection, union, target_area = intersectionAndUnion(mask, gt_mask)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target_area)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
            'class_name': class_name, 
            'IoU': iou_class, 
            'Accuracy': accuracy_class, 
            'algo': algo_name, 
            'model': model_name
        }])], ignore_index=True)
    mIoU = result_dataframe.IoU.mean()
    mAcc = result_dataframe.Accuracy.mean()
    print(f"\nDataset: {args.dataset_name}, Model: {model_name}, Algorithm: {algo_name}")
    print("nmIoU: %.2f" % (100 * mIoU))
    print("mAcc: %.2f\n" % (100 * mAcc))
    return result_dataframe, mIoU, mAcc




def main():
    args = get_arguments()
    result_dataframe = pd.DataFrame(columns=['obj_name', 'IoU', 'Accuracy'])


    if not os.path.exists('outputs/'):
        os.mkdir('./outputs/')
    if not os.path.exists(f'outputs/{args.dataset_name}'):
        os.mkdir(f'outputs/{args.dataset_name}')


    if args.sam_name == "all":
        for sam_name in MODEL_MAP.keys():
            model = load_model(sam_name)
            if args.dataset_name == "all":
                for dataset_name in DATASETS:
                    dataframe = load_dataset(dataset_name)
                    result, mIoU, mAcc = predict_on_dataset(args, model, dataframe, sam_name, args.algo)
                    result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                        'model': sam_name, 
                        'dataset': dataset_name, 
                        'mIoU': mIoU, 
                        'mAcc': mAcc, 
                        'algo': args.algo
                    }])], ignore_index=True)
            else:
                dataframe = load_dataset(args.dataset_name)
                result, mIoU, mAcc = predict_on_dataset(args, model, dataframe, sam_name, args.algo)
                result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                    'model': sam_name, 
                    'dataset': args.dataset_name, 
                    'mIoU': mIoU, 
                    'mAcc': mAcc, 
                    'algo': args.algo
                }])], ignore_index=True)
    else:
        model = load_model(args.sam_name)
        dataframe = load_dataset(args.dataset_name)
        result, mIoU, mAcc = predict_on_dataset(args, model, dataframe, args.sam_name, args.algo)
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
            'model': args.sam_name, 
            'dataset': args.dataset_name, 
            'mIoU': mIoU, 
            'mAcc': mAcc, 
            'algo': args.algo
        }])], ignore_index=True)
    print(f"\n\n{result_dataframe}")

if __name__ == "__main__":
    main()







