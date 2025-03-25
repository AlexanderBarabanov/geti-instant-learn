import cv2
from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from utils.data import load_dataset

if __name__ == "__main__":
    p = PerSam()
    per_seg_dataset = load_dataset("PerSeg")
    image_classes_filter = "can"
    dataset = per_seg_dataset[per_seg_dataset["class_name"] == image_classes_filter]
    for class_name in dataset.class_name.unique():
        class_samples = dataset[dataset["class_name"] == class_name]
        references = class_samples.head(1)
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
        a = p.infer(target_images)
        state = p.get_state()
        p.reset_state()

    print(state)
