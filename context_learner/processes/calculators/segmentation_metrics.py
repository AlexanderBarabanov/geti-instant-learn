from typing import List, Dict, Any
import numpy as np

from context_learner.processes.calculators.calculator_base import Calculator
from context_learner.types import Masks, State
from sklearn.metrics import confusion_matrix


class SegmentationMetrics(Calculator):
    def __init__(self, state: State, categories: List[str]):
        """
        Ths class handles metrics calculations.

        Args:
            state: The state object of the pipeline
            categories: A list of category names that will be used.
                Note: The background should not be present and is added automatically.
        """
        super().__init__(state)
        self.n_samples = 0
        self.tp_count = self.fp_count = self.fn_count = self.tn_count = 0
        self.confusion: Dict[str : np.ndarray] = (
            dict()
        )  # category: binary confusion summation
        if categories is not None:
            self.categories = list({"background"}.union(set(categories)))

    def get_metrics(self) -> Dict[str, List[Any]]:
        d = {
            "category": [],
            "true_positives": [],
            "true_negatives": [],
            "false_positives": [],
            "false_negatives": [],
            "precision": [],
            "recall": [],
            "f1score": [],
            "jaccard": [],
            "iou": [],
            "dice": [],
            "accuracy": [],
        }
        for cat_name, confusion in self.confusion.items():
            diag = np.diag(confusion)
            up = np.triu(confusion, k=1)
            lo = np.tril(confusion, k=1)
            tn = diag[0]
            tp = diag[1:].sum()
            fn = up.sum()
            fp = lo.sum()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            f1score = 2 * ((precision * recall) / (precision + recall))
            jaccard = tp / (fn + fp - tp)
            iou = jaccard
            dice = f1score
            d["category"].append(cat_name)
            d["true_positives"].append(tp)
            d["true_negatives"].append(tn)
            d["false_positives"].append(fp)
            d["false_negatives"].append(fn)
            d["precision"].append(precision)
            d["recall"].append(recall)
            d["f1score"].append(f1score)
            d["jaccard"].append(jaccard)
            d["iou"].append(iou)
            d["dice"].append(dice)
            d["accuracy"].append(accuracy)
        return d

    def __call__(
        self,
        predictions: List[Masks],
        references: List[Masks],
        mapping: Dict[int, str] = None,
    ):
        """
        This class compares predicted and reference masks. Individual instances are merged into one mask.
        If no mapping is provided then a mapping is created according to indices in self.categories.
        Currently, this implementation supports only binary masks per class.

        Args:
            predictions: List of predicted masks
            references: List of reference masks
            mapping: Dictionary mapping class ids to class names.
        """
        if mapping is None:
            mapping = {idx: name for idx, name in enumerate(self.categories)}
        else:
            # For internal calculations include the background class
            mapping = {idx + 1: name for idx, name in mapping.items()}
            mapping[0] = "background"
        class_ids = list(sorted(mapping.keys()))
        if len(class_ids) > 2:
            raise NotImplementedError("Multiple classes per image not yet supported.")
        class_name = mapping[class_ids[-1]]

        # Start metric calculation
        for prediction, reference in zip(predictions, references):
            # Create a mask where each pixel value represents the class id
            pred_mask = np.zeros([*reference.mask_shape])  # pred shape can be empty
            ref_mask = np.zeros([*reference.mask_shape])
            for class_id in class_ids:
                if (
                    class_id - 1 not in reference.class_ids()
                    and class_id - 1 not in prediction.class_ids()
                ) or class_id == 0:
                    continue
                if class_id - 1 in reference.class_ids():
                    ref = reference.to_numpy(class_id - 1)
                    ref_mask[np.max(ref, axis=0) > 0] = class_id

                if class_id - 1 in prediction.class_ids():
                    pred = prediction.to_numpy(class_id - 1)
                    pred_mask[np.max(pred, axis=0) > 0] = class_id

            # Calculate confusion matrix of this image
            conf = confusion_matrix(
                y_true=ref_mask.flatten(), y_pred=pred_mask.flatten(), labels=class_ids
            )
            if class_name in self.confusion.keys():
                self.confusion[class_name] = self.confusion[class_name] + conf
            else:
                self.confusion[class_name] = conf
