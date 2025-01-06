import colorsys
from functools import partial
from typing import List
import gradio as gr
import cv2
import numpy as np
import os

from algorithms import load_model, PerSamPredictor
from constants import MODEL_MAP
from model_api.models import (
    Prompt,
    Model,
    SAMLearnableVisualPrompter,
    ZSLVisualPromptingResult,
)
from model_api.models.visual_prompting import SAMPartAwareLearnableVisualPrompter


def get_colors(n: int):
    HSV_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return (np.array(list(RGB_tuples)) * 255).astype(np.uint8)


example_pairs = [
    {
        "label": "Bag Example",
        "reference": "data/PerSeg/Images/backpack/00.jpg",
        "target": "data/PerSeg/Images/backpack/01.jpg",
    },
    {
        "label": "Peanuts Example",
        "reference": "data/peanuts_coco/train/WIN_20220423_18_13_48_Pro_jpg.rf.2f7f31f6c6e102cab343008ab4f45b6f.jpg",
        "target": "data/peanuts_coco/train/WIN_20220502_18_31_01_Pro_jpg.rf.ef0644a70513801a980796f92d6046b1.jpg",
    },
    {
        "label": "Potatoes1",
        "reference": "data/Potatoes/1 1.bmp",
        "target": "data/Potatoes/2 1.bmp",
    },
    {
        "label": "Potatoes2",
        "reference": "data/Potatoes/1 1.bmp",
        "target": "data/Potatoes/8 1.bmp",
    },
    {
        "label": "Potatoes Large Scene",
        "reference": "data/Potatoes/scene00001.jpg",
        "target": "data/Potatoes/scene00121.jpg",
    },
]

os.makedirs("data", exist_ok=True)

encoder_path = "data/otx_models/sam_vit_b_zsl_encoder.xml"
decoder_path = "data/otx_models/sam_vit_b_zsl_decoder.xml"

zsl_sam_prompter: (
    SAMLearnableVisualPrompter
    | SAMPartAwareLearnableVisualPrompter
    | PerSamPredictor
    | None
) = None

clicks = []
label_mode = 0  # 0 for Label A, 1 for Label B
algo_name = "Personalized SAM"
model_name = "MobileSAM-MAPI"
point_colors = [(255, 0, 0), (0, 255, 0)]


def load_example(selected_label):
    pair = next(item for item in example_pairs if item["label"] == selected_label)
    reference_image = preload_image(pair["reference"])
    target_image = preload_image(pair["target"], name="target_img.jpg")
    return reference_image, target_image


def init_algo(selected_algo: str):
    global algo_name, model_name
    algo_name = selected_algo
    init_model(model_name)


def init_model(selected_model: str):
    global zsl_sam_prompter, algo_name, model_name
    print(f"Loading model {selected_model}")
    zsl_sam_prompter = load_model(sam_name=selected_model, algo_name=algo_name)
    print("Model loaded.")


def set_label_mode(val):
    global label_mode
    label_mode = val


def add_output_to_image(
    original_img: np.array, input_points: List[Prompt], result: ZSLVisualPromptingResult
) -> tuple[np.array, str]:
    mask_colors = get_colors(len(input_points))
    image = original_img.copy()
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
                int(image.shape[0] / 100),
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


def get_sam_output(
    point_prompts: List[Prompt],
    reference_img: np.array,
    target_img: np.array,
    apply_refinement: bool,
) -> tuple[np.array, tuple[np.array, str], List]:
    input_image = cv2.imread("data/input_img.jpg")
    reference_features, masks = zsl_sam_prompter.learn(
        reference_img, points=point_prompts
    )
    if reference_features is None or masks is None:
        return input_image, (target_img, "no reference mask"), []
    result, all_predictions = zsl_sam_prompter.infer(
        target_img, dev=True, apply_masks_refinement=apply_refinement
    )

    print(
        "shape of reference features after learning:",
        reference_features.feature_vectors.shape,
    )
    target_img_with_output = add_output_to_image(target_img, point_prompts, result)

    colors = get_colors(len(masks))
    for idx, mask in enumerate(masks):
        masked_img = np.where(mask[..., None], colors[idx], input_image)
        input_image = cv2.addWeighted(input_image, 0.5, masked_img, 0.5, 0)
    for i, prompt in enumerate(point_prompts):
        cv2.circle(
            input_image,
            prompt.data,
            int(input_image.shape[0] / 100),
            point_colors[prompt.label],
            -1,
        )

    return input_image, (target_img_with_output, "result"), []


def on_select(evt: gr.SelectData):
    global clicks, label_mode
    x, y = int(evt.index[0]), int(evt.index[1])
    print(f"Selected point at ({x}, {y})")
    original_img = cv2.imread("data/input_img.jpg")
    clicks.append(Prompt(np.array([x, y]), label_mode))
    img_with_inputs = original_img.copy()
    for prompt in clicks:
        cv2.circle(
            img_with_inputs,
            prompt.data,
            int(original_img.shape[0] / 100),
            point_colors[prompt.label],
            -1,
        )
    return img_with_inputs


def process_predictions(apply_refinement):
    global clicks
    original_img = cv2.imread("data/input_img.jpg")
    target_image = cv2.imread("data/target_img.jpg")

    if clicks:
        input_img_with_mask, (result_img, caption), gallery_list = get_sam_output(
            clicks, original_img, target_image, apply_refinement
        )
        cv2.imwrite("data/result_img.jpg", result_img)
        clicks = []
        return input_img_with_mask, result_img, gallery_list
    return original_img, None, []


def save_on_disk(img, name="input_img.jpg"):
    cv2.imwrite(f"data/{name}", img)
    return img


def preload_image(example_image_path, name="input_img.jpg"):
    img = cv2.imread(example_image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        save_on_disk(img, name)
    return img


# Preload the default example pair, model and algorithm
default_example_label = example_pairs[1]["label"]
default_reference_image, default_target_image = load_example(default_example_label)
init_algo(algo_name)

with gr.Blocks() as demo:
    title = gr.HTML("<h1>Visual Prompting Demo</h1>")
    with gr.Row():
        # Example selector dropdown
        example_selector = gr.Dropdown(
            label="Select Example Pair",
            choices=[pair["label"] for pair in example_pairs],
            value=default_example_label,  # Default selection
        )

        # model selector:
        algorithm_selector = gr.Radio(
            label="Select Algorithm",
            choices=["Personalized SAM", "Personalized Part-Aware SAM"],
            value=algo_name,
        )
        algorithm_selector.change(init_algo, inputs=algorithm_selector)
        model_selector = gr.Radio(
            label="Select backend",
            choices=MODEL_MAP.keys(),
            value=model_name,
        )
        model_selector.change(init_model, inputs=model_selector)
        label_slider = gr.Slider(
            minimum=0,
            maximum=1,
            step=1,
            value=0,
            label="Select Label",
            interactive=True,
        )
        label_slider.change(lambda val: set_label_mode(val), inputs=label_slider)

    with gr.Row():
        reference_image = gr.Image(
            type="numpy",
            label="Upload Image",
            interactive=True,
            value=default_reference_image,
        )
        target_image = gr.Image(
            type="numpy",
            label="Upload Target Image",
            interactive=True,
            value=default_target_image,
        )

    with gr.Row():
        submit_button = gr.Button("Submit")
        refinement_checkbox = gr.Checkbox(label="Apply Post Refinement", value=False)

    # Gallery component to display results
    gallery = gr.Gallery(label="Gallery of Results", columns=4)

    example_selector.change(
        load_example, inputs=example_selector, outputs=[reference_image, target_image]
    )
    reference_image.upload(
        save_on_disk, inputs=reference_image, outputs=reference_image
    )
    target_image.upload(
        partial(save_on_disk, name="target_img.jpg"),
        inputs=target_image,
        outputs=target_image,
    )
    reference_image.select(on_select, outputs=reference_image)

    submit_button.click(
        process_predictions,
        inputs=[refinement_checkbox],
        outputs=[reference_image, target_image, gallery],
    )

# Launch the Gradio UI on port 8888
demo.launch(server_port=8888, server_name="0.0.0.0")
