import numpy as np
import torch
from context_learner.types.prompts import Prompt


class Masks(Prompt):
    """
    This class represents all class masks for a single image.
    Masks are stored as a dictionary of torch tensors, where the key is the class id.
    Masks per class are stored as a 3D tensor with shape n_masks x H x W and boolean values.
    """

    def add(self, mask: torch.Tensor | np.ndarray, class_id: int = 0) -> None:
        """
        Add a mask for a given class.
        """
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if not mask.dtype == torch.bool:
            mask = (mask > 0).bool()

        if mask.ndim == 3 and mask.shape[0] != 1:  # HWC format
            if mask.shape[-1] == 1:
                max_channel = 0
            else:
                max_channel = torch.argmax(mask.sum(dim=(0, 1)))
            mask = mask[:, :, max_channel].unsqueeze(0)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if class_id not in self._data:
            self._data[class_id] = mask
        else:
            # Concatenate along existing batch dimension (dim=0) without adding new dimension
            self._data[class_id] = torch.cat([self._data[class_id], mask], dim=0)

    def resize(self, size: tuple[int, int]) -> "Masks":
        """
        Return a resized copy of the masks.
        """
        resized_masks = Masks()
        for class_id in self._data:
            resized_masks.add(
                torch.nn.functional.interpolate(
                    self._data[class_id], size=size, mode="bilinear"
                ),
                class_id,
            )
        return resized_masks

    def resize_inplace(self, size: tuple[int, int]) -> None:
        """
        Resize the masks in place.
        """
        for class_id in self._data:
            self._data[class_id] = torch.nn.functional.interpolate(
                self._data[class_id], size=size, mode="bilinear"
            )

    def to_numpy(self, class_id: int = 0) -> np.ndarray:
        """
        Convert the masks to a numpy array with shape HxWxC in uint8 format.
        """
        return self._data[class_id].numpy().astype(np.uint8)

    @property
    def mask_shape(self) -> tuple[int, int]:
        """
        Get the shape of a mask.
        """
        return self._data[0].shape[1:]
