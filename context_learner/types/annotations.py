from typing import List
from context_learner.types.data import Data


class Annotations(Data):
    """
    This class represents annotations for a single image. Annotations are the datatype that Geti expects.
    """

    def __init__(self):
        self.polygons = {}

    def add_polygon(self, points: List[List[float]], class_id: int = 0) -> None:
        """
        Add a polygon for a given class.
        Args:
            points: List of [x, y] coordinates defining the polygon
            class_id: Class ID for this polygon
        """
        if class_id not in self.polygons:
            self.polygons[class_id] = []
        self.polygons[class_id].append(points)
