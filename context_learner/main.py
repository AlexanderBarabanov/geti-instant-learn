from typing import List
import cv2
import os
from context_learner.pipelines.matcher_pipeline import Matcher
from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from utils.data import load_dataset
from context_learner.utils.visualize import overlay_masks_and_points

if __name__ == "__main__":
    p = Matcher()
    per_seg_dataset = load_dataset("PerSeg")
    image_classes_filter = "can"
    n_shot = 2
    dataset = per_seg_dataset[per_seg_dataset["class_name"] == image_classes_filter]

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for class_name in dataset.class_name.unique():
        class_samples = dataset[dataset["class_name"] == class_name]
        references = class_samples.head(n_shot)
        targets = class_samples[~class_samples.index.isin(references.index)]

        reference_images = []
        reference_priors = []
        for _, reference in references.iterrows():
            image_np = cv2.imread(reference.image)
            mask_np = cv2.imread(reference.mask_image)

            reference_images.append(Image(image_np))
            masks = Masks()
            masks.add(mask_np)
            reference_priors.append(Priors(masks=masks))

        p.learn(reference_images=reference_images, reference_priors=reference_priors)

        target_images = [
            Image(cv2.imread(target.image)) for _, target in targets.iterrows()
        ]
        # Capture masks and used_points
        p.infer(target_images)

        # Visualize and save results
        for i, (target_image, masks, points) in enumerate(
            zip(p._state.target_images, p._state.masks, p._state.used_points)
        ):
            output_image = overlay_masks_and_points(target_image, masks, points)
            save_path = os.path.join(results_dir, f"{class_name}_target_{i}_result.png")
            cv2.imwrite(save_path, output_image)
            print(f"Saved visualization to: {save_path}")

        state = p.get_state()
        p.reset_state()

    print(state)
