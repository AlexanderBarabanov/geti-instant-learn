# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification example."""

import torch
from argparse import ArgumentParser
from getiprompt.pipelines.dinotxt import DinoTxtZeroShotClassification
from getiprompt.types import Priors
from pathlib import Path
from datumaro import Dataset
import cv2
from getiprompt.utils import precision_to_torch_dtype


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=24)

    args = parser.parse_args()
    
    # parse arguments
    data_root = Path(args.data_root)
    subset = args.subset
    batch_size = args.batch_size
    precision = precision_to_torch_dtype(args.precision)


    # import dataset
    dataset = Dataset.import_from(
        path=data_root,
        format="imagenet_with_subset_dirs"
    )
    label_names = dataset.get_label_cat_names()    

    # select subset, default is all
    if subset != "all":
        dataset = dataset.subsets()[subset]
    
    # load target images and gt labels
    target_images = []
    gt_labels = []
    for item in dataset:
        target_images.append(cv2.cvtColor(item.media.data, cv2.COLOR_BGR2RGB))
        gt_labels.append(item.annotations[0].label)

    # initialize DinoTxt pipeline
    dinotxt = DinoTxtZeroShotClassification(
        precision=precision,
        pretrained=True,
    )

    # learn from text prompts
    dinotxt.learn(
        reference_images=[], 
        reference_priors=[Priors(text={i: label_name for i, label_name in enumerate(label_names)})],
    )

    # infer
    pred_labels = []
    for i in range(0, len(target_images), batch_size):
        chunk = target_images[i : i + batch_size]
        results = dinotxt.infer(target_images=chunk)
        for mask in results.masks:
            pred_labels.append(mask.class_ids()[0])
    pred_labels = torch.stack(pred_labels).cuda()
    gt_labels = torch.tensor(gt_labels).cuda()
    
    # calculate zero-shot classification accuracy
    accuracy = sum(pred_labels == gt_labels) / len(gt_labels)
    print(f"Accuracy: {accuracy}")
