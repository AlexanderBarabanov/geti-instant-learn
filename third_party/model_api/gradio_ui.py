from functools import partial
from typing import List

import gradio as gr
import cv2
import numpy as np
import os

from examples.python.zsl_visual_prompting.run import get_colors
from model_api.models import Prompt, Model, SAMLearnableVisualPrompter, ZSLVisualPromptingResult
from model_api.models.visual_prompting import SAMPartAwareLearnableVisualPrompter

os.makedirs("data", exist_ok=True)
# example_reference_image_path = "data/aerial1.jpg"
# example_target_image_path = "data/aerial2.jpg"
# example_reference_image_path = "data/horse1.jpeg"
# example_target_image_path = "data/horse2.jpeg"
example_reference_image_path = "data/bag1.jpg"
example_target_image_path = "data/bag2.jpg"

encoder_path = "data/otx_models/sam_vit_b_zsl_encoder.xml"
decoder_path = "data/otx_models/sam_vit_b_zsl_decoder.xml"

encoder = Model.create_model(encoder_path)
decoder = Model.create_model(decoder_path)
zsl_sam_prompter: SAMLearnableVisualPrompter | SAMPartAwareLearnableVisualPrompter | None = None

# Global store for click points
clicks = []

# Global variable to track "two objects" checkbox state
sam_model_name = "Personalized Part-Aware SAM"
two_objects_mode = False
multiple_prompt_mode = False


def init_SAM_prompter(selected_model: str):
    global zsl_sam_prompter, sam_model_name
    print("Loading selected model:", selected_model)
    if selected_model == "Personalized SAM":
        zsl_sam_prompter = SAMLearnableVisualPrompter(encoder, decoder)
    elif selected_model == "Personalized Part-Aware SAM":
        zsl_sam_prompter = SAMPartAwareLearnableVisualPrompter(encoder, decoder)
    sam_model_name = selected_model


def toggle_two_objects(selected: bool):
    global two_objects_mode
    two_objects_mode = selected
    # Reset clicks when mode changes
    clicks.clear()


def toggle_multiple_prompt(selected: bool):
    global multiple_prompt_mode
    multiple_prompt_mode = selected
    # Reset clicks when mode changes
    clicks.clear()


def add_output_to_image(
    original_img: np.array, input_points: List[Prompt], result: ZSLVisualPromptingResult
) -> tuple[np.array, str]:
    mask_colors = get_colors(len(input_points))
    point_colors = [(255, 0, 0), (0, 255, 0)]

    image = original_img.copy()

    # circle radius should relative to image size otherwise its too small for large images
    relative_circle_radius = int(image.shape[0] / 100)

    # show the points and masks on the target image
    for i in result.data:
        masks = result.get_mask(i)
        for j, instance in enumerate(masks.mask):
            masked_img = np.where(instance[..., None], mask_colors[i], image)
            image = cv2.addWeighted(image, 0.2, masked_img, 0.8, 0)
    for i in result.data:
        masks = result.get_mask(i)
        # add points and scores after all masks are placed
        for j, point in enumerate(masks.points):
            confidence = float(masks.scores[j])
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                relative_circle_radius,
                point_colors[i],
                -1,
            )
            cv2.putText(
                image,
                f"{confidence:.3f}",
                (int(point[0]), int(point[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                point_colors[i],
                2,
                cv2.LINE_AA,
            )
    return image


def all_predictions_per_label_to_gallery(original_img: np.array, all_predictions: dict) -> List[tuple[np.array, str]]:
    mask_colors = get_colors(len(all_predictions))
    point_colors = [(255, 0, 0), (0, 255, 0)]
    relative_circle_radius = int(original_img.shape[0] / 100)

    gallery = []
    for label, predictions in all_predictions.items():
        for prediction in predictions:
            image = original_img.copy()
            caption = f"{prediction['num_clusters']} clusters. Distribution distance: {prediction['distribution_distance']:.3f}"
            points = prediction["point_prompts"]
            masks = prediction["upscaled_masks"]
            # add mask to the image
            masked_img = np.where(masks[0][..., None], mask_colors[label], image)
            image = cv2.addWeighted(image, 0.2, masked_img, 0.8, 0)
            # add points
            for j, point in enumerate(points):
                cv2.circle(
                    image,
                    (int(point[0]), int(point[1])),
                    relative_circle_radius,
                    point_colors[label],
                    -1,
                )
            gallery.append((image.copy(), caption))
    return gallery


def get_sam_output(point_prompts: List[Prompt], reference_img: np.array, target_img: np.array) -> tuple[np.array, list]:
    zsl_sam_prompter.learn(reference_img, points=point_prompts)
    result, all_predictions = zsl_sam_prompter.infer(target_img, dev=True)

    target_img_with_output = add_output_to_image(target_img, point_prompts, result)

    if all_predictions:
        gallery = all_predictions_per_label_to_gallery(target_img, all_predictions)
        caption = f"using {all_predictions} part-level features"
        return target_img_with_output, gallery

    return (target_img_with_output, "result"), []


def on_select(evt: gr.SelectData):
    global clicks, two_objects_mode
    x, y = int(evt.index[0]), int(evt.index[1])
    print(f"Selected point at ({x}, {y})")
    original_img = cv2.imread("data/input_img.jpg")
    target_image = cv2.imread("data/target_img.jpg")
    clicks.append((x, y))

    if not two_objects_mode or len(clicks) == 1:
        # Single click mode or first click in two-object mode
        img_with_dot = original_img.copy()
        cv2.circle(img_with_dot, (x, y), radius=original_img.shape[0] // 100, color=(255, 0, 0), thickness=-1)

        if not two_objects_mode:  # Directly process for single-object mode
            prompt_points = [Prompt(np.array(click), idx) for idx, click in enumerate(clicks)]
            (result_img, caption), gallery_list = get_sam_output(prompt_points, original_img, target_image)

            # Save result image
            cv2.imwrite("data/result_img.jpg", result_img)
            clicks = []
            return img_with_dot, result_img, gallery_list

        return img_with_dot, (target_image, ), []

    elif len(clicks) == 2:
        # Two-object mode: process after second click
        img_with_dot = original_img.copy()
        cv2.circle(img_with_dot, (clicks[0][0], clicks[0][1]), radius=5, color=(255, 0, 0), thickness=-1)  # Red dot
        cv2.circle(img_with_dot, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot

        # Create prompt points and process SAM output
        prompt_points = [
            Prompt(np.array(click), 0 if multiple_prompt_mode else idx) for idx, click in enumerate(clicks)
        ]
        result_img, gallery_list = get_sam_output(prompt_points, original_img, target_image)

        # Save result image
        cv2.imwrite("data/result_img.jpg", result_img)
        clicks = []
        return img_with_dot, result_img, gallery_list


def save_on_disk(img, name="input_img.jpg"):
    cv2.imwrite(f"data/{name}", img)
    return img


def preload_image(example_image_path, name="input_img.jpg"):
    img = cv2.imread(example_image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        save_on_disk(img, name)
    return img


## Load initial values/defaults and start UI.

example_reference_image = preload_image(example_reference_image_path)
example_target_image = preload_image(example_target_image_path, name="target_img.jpg")
init_SAM_prompter(sam_model_name)

with gr.Blocks() as demo:
    title = gr.HTML("<h1>Visual Prompting Demo</h1>")
    with gr.Row():
        # model selector:
        sam_model_selector = gr.Radio(
            label="Select Model",
            choices=["Personalized SAM", "Personalized Part-Aware SAM"],
            value=sam_model_name,
        )
        sam_model_selector.change(init_SAM_prompter, inputs=sam_model_selector)
        two_objects_checkbox = gr.Checkbox(label="Two Objects", value=False)
        two_objects_checkbox.change(toggle_two_objects, inputs=two_objects_checkbox)
        multiple_prompt_checkbox = gr.Checkbox(label="Multiple Prompts", value=False)
        multiple_prompt_checkbox.change(toggle_multiple_prompt, inputs=multiple_prompt_checkbox)

    with gr.Row():
        reference_image = gr.Image(type="numpy", label="Upload Image", interactive=True, value=example_reference_image)
        target_image = gr.Image(type="numpy", label="Upload Target Image", interactive=True, value=example_target_image)

    # Gallery component to display results
    gallery = gr.Gallery(label="Gallery of Results", columns=4)

    reference_image.upload(save_on_disk, inputs=reference_image, outputs=reference_image)
    target_image.upload(partial(save_on_disk, name="target_img.jpg"), inputs=target_image, outputs=target_image)
    reference_image.select(on_select, outputs=[reference_image, target_image, gallery])


# Launch the Gradio UI on port 8888
demo.launch(server_port=8888)
