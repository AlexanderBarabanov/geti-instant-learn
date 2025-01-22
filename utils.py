import colorsys
import os
from typing import List, Dict

import numpy as np
import ot
import pandas as pd
import cv2

from model_api.models import Prompt
from constants import DATA_PATH

DATAFRAME_COLUMNS = ['class_name', 'file_name', 'image', 'mask_image', 'frame']

def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "PerSeg":
        return load_perseg_data()
    elif dataset_name == "DAVIS":
        return load_davis_data()


def load_perseg_data() -> pd.DataFrame:
    images_path = os.path.join(DATA_PATH, 'PerSeg', 'Images')
    annotations_path = os.path.join(DATA_PATH, 'PerSeg', 'Annotations')

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    for class_name in os.listdir(images_path):
        if ".DS" in class_name:
            continue

        for file_name in os.listdir(os.path.join(images_path, class_name)):
            if ".DS" in file_name:
                continue

            frame = int(file_name[:-4])  # Remove .jpg and convert to int

            data = pd.concat([data, pd.DataFrame([{
                'class_name': class_name,
                'file_name': file_name,
                'image': os.path.join(images_path, class_name, file_name),
                'mask_image': os.path.join(annotations_path, class_name, file_name[:-4] + '.png'),
                'frame': frame
            }])], ignore_index=True)

    # sort on class_name and frame
    data.sort_values(by=['class_name', 'frame'], inplace=True)
    return data



def load_davis_data() -> pd.DataFrame:
    """Load DAVIS dataset into a pandas DataFrame.
    Returns DataFrame with columns: class_name, file_name, image, mask_image, frame
    """
    images_path = os.path.join(DATA_PATH, 'DAVIS' 'JPEGImages', '480p')
    annotations_path = os.path.join(DATA_PATH, 'DAVIS', 'Annotations', '480p')
    imagesets_path = os.path.join(DATA_PATH, 'DAVIS', 'ImageSets', '2017','val.txt')

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    with open(imagesets_path, "r") as f:
        sequences = [x.strip() for x in f.readlines()]

    for sequence in sequences:
        frames = sorted(os.listdir(os.path.join(images_path, sequence)))
        
        for frame in frames:
            if frame.endswith('.jpg'):
                frame_id = frame[:-4]  # Remove .jpg extension
                mask_file = frame_id + '.png'
                
                frame_number = int(frame_id)
                
                data = pd.concat([data, pd.DataFrame([{
                    'class_name': sequence,
                    'file_name': frame,
                    'image': os.path.join(images_path, sequence, frame),
                    'mask_image': os.path.join(annotations_path, sequence, mask_file),
                    'frame': frame_number
                }])], ignore_index=True)

    # Sort by class_name and frame
    data.sort_values(by=['class_name', 'frame'], inplace=True)
    return data


def mask_image_to_polygon_prompts(mask_image: np.array) -> List[Prompt]:
    mask = mask_image.astype(np.uint8)
    mask = np.stack((mask, np.zeros_like(mask), np.zeros_like(mask)), axis=-1)




def get_colors(n: int):
    HSV_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return (np.array(list(RGB_tuples)) * 255).astype(np.uint8)

def transform_point_prompts_to_dict(prompts: List[Prompt]) -> Dict[int, List[List[int]]]:
    result = {}
    for prompt in prompts:
        label = prompt.label
        coords = prompt.data.tolist()
        if label not in result:
            result[label] = []
        result[label].append(coords)
    return result

def transform_mask_prompts_to_dict(prompts: List[Prompt]) -> Dict[int, np.array]:
    """
    Transform masks into a dictionary of masks per class.
    
    Args:
        prompts: List[Prompt]
    Returns:
        Dict[int, np.array]  where the key is the class index and the value is the mask 
    """
    result = {}
    for prompt in prompts:
        label = prompt.label
        mask = prompt.data
        result[label] = mask
    return result

def save_visualization(image: np.ndarray, mask, output_path: str, points=None, scores=None) -> None:
    """
    Save a visualization of the segmentation mask overlaid on the image.
    
    Args:
        image: RGB image as numpy array
        mask: Segmentation mask object with mask.mask containing instance masks
        output_path: Path where to save the visualization
        points: Optional points to visualize
        scores: Optional confidence scores for the points
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get unique colors for each instance mask
    mask_colors = get_colors(len(mask.mask))
    image_vis = image.copy()
    
    # Draw each instance mask with a different color
    for i, instance in enumerate(mask.mask):
        masked_img = np.where(instance[..., None], mask_colors[i], image_vis)
        image_vis = cv2.addWeighted(image_vis, 0.2, masked_img, 0.8, 0)
    
    # Draw points and confidence scores if provided
    if points is not None and scores is not None:
        for i, point in enumerate(points):
            # Draw star marker
            x, y = int(point[0]), int(point[1])
            size = int(image.shape[0] / 50)  # Scale marker size with image
            cv2.drawMarker(image_vis, (x, y), (255, 255, 255), cv2.MARKER_STAR, size)
            
            # Add confidence score text
            confidence = float(scores[i])
            cv2.putText(image_vis, 
                      f"{confidence:.2f}", 
                      (x + 5, y - 5),  # Offset text slightly from point
                      cv2.FONT_HERSHEY_SIMPLEX,
                      image.shape[0] / 1500,  # Font scale relative to image height
                      (255, 255, 255),  # White text
                      1)  # Line thickness

    # Save visualization
    cv2.imwrite(output_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)) 




def _compute_wasserstein_distance(a, b, weights=None) -> float:
    """
    Computes the Wasserstein distance between two distributions a and b.
    Lower distance is better match.
    :param a:
    :param b:
    :param weights:
    :return:
    """
    n_a = a.shape[0]
    a_hist = ot.unif(n_a)
    if weights is not None:
        b_hist = weights / np.sum(weights)
    else:
        n_b = b.shape[0]
        b_hist = ot.unif(n_b)

    M = ot.dist(a, b)
    wasserstein_distance = ot.emd2(a_hist, b_hist, M, numItermax=10000000)
    return wasserstein_distance