import torch

from context_learner.types.data import Data


class Features(Data):

    """
    This class represents features.
    """

    def __init__(self, shape: torch.Size = None):
        self._features = None
        self._shape = shape

    def from_tensor(self, features: torch.Tensor):
        """
        Initialize this class from a torch tensor.

        The input feature shape should adhere to the predefined shape.

        Args:
            features: A tensor containing the embedding of an image.

        Returns:
            None

        Examples:
            >>> t = torch.tensor([[[0, 1],
            ...                    [2, 3]]])
            >>> feat = Features(torch.Size([2, 2]))
            >>> feat.from_tensor(t)
            >>> t2 = torch.tensor([0, 1])
            >>> feat.from_tensor(t2)
            Traceback (most recent call last):
            ...
            AssertionError
        """

        assert self._shape is None or features.shape[-len(self._shape):] == self._shape
        self._features = features
