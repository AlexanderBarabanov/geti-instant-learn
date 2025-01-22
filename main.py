import argparse
import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import PerSamPredictor, load_model
from constants import *
from P2SAM.eval_utils import AverageMeter, intersectionAndUnion
from model_api.models.result_types.visual_prompting import ZSLVisualPromptingResult
from model_api.models.visual_prompting import Prompt, SAMLearnableVisualPrompter
from utils import load_dataset, save_visualization



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam_name', type=str, default="MobileSAM", choices=MODEL_MAP.keys())
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--algo',type=str, default='persam', choices=ALGORITHMS)
    parser.add_argument("--num_priors", type=int, default=1, help="Number of prior images to use as references")
    parser.add_argument("--dataset_name", type=str, default="PerSeg", choices=DATASETS)
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--show", action="store_true", help="Show results during processing")
    parser.add_argument("--post_refinement", action="store_true", help="Apply post refinement")
    parser.add_argument("--target_guided_attention", action="store_true", help="Use target guided attention for the SAM model. This passes the target similarity matrix and reference features to the decoder")
    parser.add_argument("--class_name", type=str, default=None, help="Filter on class name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output data")
    parser.add_argument("--num_clusters", type=int, default=1, help="Number of clusters for PartAwareSAM, if 1 use mean of all features")
    args = parser.parse_args()
    return args

def predict_on_dataset(args: argparse.Namespace, predictor: PerSamPredictor | SAMLearnableVisualPrompter, dataframe: pd.DataFrame, model_name: str, algo_name: str, output_path: str) -> tuple[float, float]:
    result_dataframe = pd.DataFrame(columns=['class_name', 'IoU', 'Accuracy'])
    print(f"Output path: {output_path}")
    
    if os.path.exists(output_path):
        if args.overwrite:
            shutil.rmtree(output_path)
        else:
            choice = input(f"Output path {output_path} already exists. Do you want to overwrite it? (y/n)\n")
            if choice == "y":
                print("Removing existing output data")
                shutil.rmtree(output_path)
            else:
                output_path += "_1"
                print(f"Output path set to {output_path}")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if args.class_name:  # filter on class_name
        dataframe = dataframe[dataframe.class_name == args.class_name]  

    for class_name in tqdm(dataframe.class_name.unique(), desc="Processing classes", total=len(dataframe.class_name.unique()), position=0, leave=True):
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        inference_time_meter = AverageMeter()

        class_samples = dataframe[dataframe.class_name == class_name]
        priors = class_samples.head(args.num_priors)
        # select remaining images as target images but do not change the order of the dataframe 
        targets = class_samples[~class_samples.index.isin(priors.index)]

        # Load all prior images and masks
        prior_images = []
        prior_masks = []
        for _, prior in priors.iterrows():
            # load image from disk and convert to numpy array
            image = cv2.cvtColor(cv2.imread(prior.image), cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(cv2.imread(prior.mask_image), cv2.COLOR_BGR2RGB)
            mask = mask_image[:,:,0]  # only need a grid, not a 3d color img. 
            mask_prompt = Prompt(label=0, data=mask_image)  # TODO only single class per image is supported for now
            prior_images.append(image)
            prior_masks.append(mask_prompt)

        if args.save:
            # save prior images and masks to disk, on top of each other
            os.makedirs(os.path.join(output_path, 'predictions', class_name), exist_ok=True)
            for i, (image, mask) in enumerate(zip(prior_images, prior_masks)):
                mask = np.where(mask.data > 0, 255, mask.data)  # for better visualization
                overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{output_path}/predictions/{class_name}/prior_{i}.png", overlay)

        # TODO currently multiple priors is not yet implemented
        predictor.learn(image=prior_images[0], masks=prior_masks, show=args.show, num_clusters=args.num_clusters)

        # predict on target images
        for row_idx, target in tqdm(targets.iterrows(), desc="Processing samples", total=len(targets), position=1, leave=False):
            # load image from disk and convert to numpy array
            target_image = cv2.cvtColor(cv2.imread(target.image), cv2.COLOR_BGR2RGB)
            gt_mask = cv2.cvtColor(cv2.imread(target.mask_image), cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            result: ZSLVisualPromptingResult = predictor.infer(image=target_image, apply_masks_refinement=args.post_refinement, target_guided_attention=args.target_guided_attention)
            inference_time_meter.update(time.time() - start_time)

            mask = result.get_mask(0)
            # Merge all instance masks into one mask using logical OR
            merged_mask = np.zeros_like(mask.mask[0], dtype=bool)
            for instance in mask.mask:
                merged_mask = np.logical_or(merged_mask, instance)

            if args.save:
                save_visualization(
                    image=target_image,
                    mask=mask,
                    output_path=os.path.join(output_path, 'predictions', class_name, os.path.basename(target.image)),
                    points=mask.points if hasattr(mask, 'points') else None,
                    scores=mask.scores if hasattr(mask, 'scores') else None
                )
            # Metrics
            mask = np.uint8(merged_mask)
            gt_mask = np.uint8(gt_mask[:,:,0] > 0)
            intersection, union, target_area = intersectionAndUnion(mask, gt_mask)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target_area)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        inference_time = inference_time_meter.sum / len(targets)
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
            'class_name': class_name, 
            'IoU': iou_class, 
            'Accuracy': accuracy_class, 
            'algo': algo_name, 
            'model': model_name,
            'inference_time': inference_time
        }])], ignore_index=True)
    mIoU = result_dataframe.IoU.mean()
    mAcc = result_dataframe.Accuracy.mean()
    inference_time = result_dataframe.inference_time.mean() 
    print(f"\nDataset: {args.dataset_name}, Model: {model_name}, Algorithm: {algo_name}")
    print("nmIoU: %.2f" % (100 * mIoU))
    print("mAcc: %.2f" % (100 * mAcc))
    print("Inference time: %.2f seconds/target image (mean per sample)" % inference_time)
    if args.save:
        result_dataframe.to_csv(f"{output_path}/results_per_class.csv", index=False)
    return mIoU, mAcc, inference_time




def main():
    args = get_arguments()
    result_dataframe = pd.DataFrame(columns=['model','algo', 'dataset', 'mIoU', 'mAcc'])

    if not os.path.exists('outputs/'):
        os.mkdir('./outputs/')
    if not os.path.exists(f'outputs/{args.dataset_name}'):
        os.mkdir(f'outputs/{args.dataset_name}')

    # Determine which models, datasets and algorithms to process
    models_to_run = MODEL_MAP.keys() if args.sam_name == "all" else [args.sam_name]
    datasets_to_run = DATASETS if args.dataset_name == "all" else [args.dataset_name]
    algorithms_to_run = ALGORITHMS if args.algo == "all" else [args.algo]

    # Process each combination
    for sam_name in models_to_run:
        for dataset_name in datasets_to_run:
            dataframe = load_dataset(dataset_name)
            for algo in algorithms_to_run:
                predictor = load_model(sam_name, algo)
                mIoU, mAcc, inference_time = predict_on_dataset(args, predictor, dataframe, sam_name, algo, f"outputs/{dataset_name}_{sam_name}_{algo}")

                result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                    'model': sam_name,
                    'algo': algo,
                    'dataset': dataset_name,
                    'mIoU': mIoU,
                    'mAcc': mAcc,
                    'inference_time': inference_time
                }])], ignore_index=True)

    print(f"\n\n{result_dataframe}")

    # save as csv
    result_dataframe.to_csv(f"outputs/{args.dataset_name}_dataset_{args.sam_name}_model_{args.algo}_results.csv", index=False)    

if __name__ == "__main__":
    main()







