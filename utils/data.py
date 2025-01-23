import os
import pandas as pd

from utils.constants import DATA_PATH, DATAFRAME_COLUMNS


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "PerSeg":
        return load_perseg_data()
    elif dataset_name == "DAVIS":
        return load_davis_data()


def load_perseg_data() -> pd.DataFrame:
    images_path = os.path.join(DATA_PATH, "PerSeg", "Images")
    annotations_path = os.path.join(DATA_PATH, "PerSeg", "Annotations")

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    for class_name in os.listdir(images_path):
        if ".DS" in class_name:
            continue

        for file_name in os.listdir(os.path.join(images_path, class_name)):
            if ".DS" in file_name:
                continue

            frame = int(file_name[:-4])  # Remove .jpg and convert to int

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [
                            {
                                "class_name": class_name,
                                "file_name": file_name,
                                "image": os.path.join(
                                    images_path, class_name, file_name
                                ),
                                "mask_image": os.path.join(
                                    annotations_path,
                                    class_name,
                                    file_name[:-4] + ".png",
                                ),
                                "frame": frame,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # sort on class_name and frame
    data.sort_values(by=["class_name", "frame"], inplace=True)
    return data


def load_davis_data() -> pd.DataFrame:
    """Load DAVIS dataset into a pandas DataFrame.
    Returns DataFrame with columns: class_name, file_name, image, mask_image, frame
    """
    images_path = os.path.join(DATA_PATH, "DAVIS" "JPEGImages", "480p")
    annotations_path = os.path.join(DATA_PATH, "DAVIS", "Annotations", "480p")
    imagesets_path = os.path.join(DATA_PATH, "DAVIS", "ImageSets", "2017", "val.txt")

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    with open(imagesets_path, "r") as f:
        sequences = [x.strip() for x in f.readlines()]

    for sequence in sequences:
        frames = sorted(os.listdir(os.path.join(images_path, sequence)))

        for frame in frames:
            if frame.endswith(".jpg"):
                frame_id = frame[:-4]  # Remove .jpg extension
                mask_file = frame_id + ".png"

                frame_number = int(frame_id)

                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            [
                                {
                                    "class_name": sequence,
                                    "file_name": frame,
                                    "image": os.path.join(images_path, sequence, frame),
                                    "mask_image": os.path.join(
                                        annotations_path, sequence, mask_file
                                    ),
                                    "frame": frame_number,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    # Sort by class_name and frame
    data.sort_values(by=["class_name", "frame"], inplace=True)
    return data
