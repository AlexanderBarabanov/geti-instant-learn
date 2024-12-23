import argparse
import os
import pandas as pd
from tqdm import tqdm
from PersonalizeSAM.per_segment_anything import SamPredictor, sam_model_registry
from algorithms import run_per_segment_anything, run_p2sam, load_model
from constants import *
from P2SAM.eval_utils import AverageMeter
from utils import load_dataset

import datumaro as dm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam_name', type=str, default="SAM",)
    parser.add_argument('--max-num-pos', type=int, default=1)
    parser.add_argument('--min-num-pos', type=int, default=1)
    parser.add_argument('--algo',type=str, default='persam', choices=['persam', 'p2sam'])
    parser.add_argument("--dataset_name", type=str, default="PerSeg", choices=DATASETS)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--post_refinement", type=bool, default=True)
    parser.add_argument("--class_name", type=str, default=None)

    args = parser.parse_args()
    return args

def predict_on_dataset(args, model):
    result_dataframe = pd.DataFrame(columns=['obj_name', 'IoU', 'Accuracy'])
    output_path = os.path.join('outputs/', args.dataset_name)
    dataframe = load_dataset(args.dataset_name)

    if args.class_name:
        dataframe = dataframe[dataframe.obj_name == args.class_name]

    for class_name in tqdm(dataframe.obj_name.unique(), desc="Processing classes", total=len(dataframe.obj_name.unique()), position=0, leave=True):
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        class_samples = dataframe[dataframe.obj_name == class_name]
        for row_idx, sample in tqdm(class_samples.iterrows(), desc="Processing samples", total=len(class_samples), position=1, leave=False):
            if args.algo == 'persam':
                fig, intersection, union, target_area = run_per_segment_anything(model, sample, output_path, save=args.save, show=args.show)
            elif args.algo == 'p2sam':
                fig, intersection, union, target_area = run_p2sam(model, sample, output_path, post_refinement=args.post_refinement, show=args.show, save=args.save)

            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_area)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{'obj_name': class_name, 'IoU': iou_class, 'Accuracy': accuracy_class, 'algo': args.algo, 'model': args.sam_name}])], ignore_index=True)
    mIoU = result_dataframe.IoU.mean()
    mAcc = result_dataframe.Accuracy.mean()
    print("\nmIoU: %.2f" % (100 * mIoU))
    print("mAcc: %.2f\n" % (100 * mAcc))
    return result_dataframe




def main():
    args = get_arguments()
    result_dataframe = pd.DataFrame(columns=['obj_name', 'IoU', 'Accuracy'])


    if not os.path.exists('outputs/'):
        os.mkdir('./outputs/')
    if not os.path.exists(f'outputs/{args.dataset_name}'):
        os.mkdir(f'outputs/{args.dataset_name}')


    if args.sam_name == "all":
        for sam_name in sam_model_registry.keys():
            model = load_model(sam_name)
            result_dataframe = pd.concat([result_dataframe, predict_on_dataset(args, model)], ignore_index=True)
        print(result_dataframe)
    else:
        model = load_model(args.sam_name)
        result_dataframe = predict_on_dataset(args, model)

if __name__ == "__main__":
    dm.Dataset.import_from("data/PerSeg_VOC", format="voc_segmentation")







