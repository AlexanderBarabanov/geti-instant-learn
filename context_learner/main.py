from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.types.annotations import Annotations
from context_learner.types.image import Image

if __name__ == "__main__":
    p = PerSam()
    p.learn([Image()] * 3, [Annotations()] * 3)
    a = p.infer([Image()])

    # Print the state
    print(p.get_state())
