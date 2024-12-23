import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PersonalizeSAM.per_segment_anything import SamPredictor, sam_model_registry
from algorithms import PerSamPredictor, run_per_segment_anything, run_p2sam, load_model
from constants import *
from P2SAM.eval_utils import AverageMeter, intersectionAndUnion
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from model_api.models.result_types.visual_prompting import ZSLVisualPromptingResult
from model_api.models.visual_prompting import SAMLearnableVisualPrompter
from utils import load_dataset

import datumaro as dm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam_name', type=str, default="SAM", choices=MODEL_MAP.keys())
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--algo',type=str, default='persam', choices=['persam', 'p2sam'])
    parser.add_argument("--num_priors", type=int, default=1, help="Number of prior images to use as references")
    parser.add_argument("--dataset_name", type=str, default="PerSeg", choices=DATASETS)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--post_refinement", type=bool, default=True)
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
        priors = class_samples.sample(n=args.num_priors, replace=False) 
        # select remaining images as target images
        targets = class_samples[~class_samples.index.isin(priors.index)]

        # learn on prior images
        # TODO modelAPI implementation does not accept masks directly. Either convert to polygons or change the implementation
        if isinstance(predictor, SAMLearnableVisualPrompter):
            raise NotImplementedError("MAPI implementation does not accept masks directly. Either convert to polygons or change the implementation")    
        predictor.learn(image = priors.image.to_numpy(), masks = priors.mask_image.tolist(), show=args.show)

        # predict on target images
        for row_idx, target in tqdm(targets.iterrows(), desc="Processing samples", total=len(targets), position=1, leave=False):
            result: ZSLVisualPromptingResult = predictor.predict(image = target.image.to_numpy(), show=args.show, apply_masks_refinement=args.post_refinement)
            mask = result.get_mask(0)
            mask = np.uint8(mask > 0)
            gt_mask = target.mask_image.to_numpy()
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







