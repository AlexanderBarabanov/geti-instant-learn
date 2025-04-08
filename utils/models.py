import os
from algorithms import PerDinoPredictor, PerSamPredictor
from model_api.models.model import Model
from model_api.models.visual_prompting import SAMLearnableVisualPrompter

from context_learner.pipelines.matcher_pipeline import Matcher
from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.pipelines.pipeline_base import Pipeline
from utils.constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH, MODEL_MAP
from PersonalizeSAM.per_segment_anything import sam_model_registry
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from PersonalizeSAM.per_segment_anything import SamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def load_model(
    backbone_name="SAM", pipeline_name="PerSAM"
) -> Pipeline:
    if backbone_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {backbone_name}")

    # Construct backbone
    name, checkpoint_path = MODEL_MAP[backbone_name]
    if backbone_name in ["SAM", "MobileSAM"]:
        sam_model = sam_model_registry[name](checkpoint=checkpoint_path).cuda()
        sam_model.eval()
        sam_model = SamPredictor(sam_model)
    elif backbone_name == "EfficientViT-SAM":
        sam_model = create_efficientvit_sam_model(
            name=name, weight_url=checkpoint_path
        ).cuda()
        sam_model.eval()
        sam_model = EfficientViTSamPredictor(sam_model)
    else:
        raise NotImplementedError(f"Model {backbone_name} not implemented yet")

    # Construct pipeline
    if pipeline_name == "PerSAMModular":
        return PerSam(sam_model)
    elif pipeline_name == "MatcherModular":
        return Matcher(sam_model)
    elif pipeline_name == "PerSAMMAPI":
        # Only MobileSAM, no backbone choice
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        return SAMLearnableVisualPrompter(encoder, decoder)
    else:
        raise NotImplementedError(f"Algorithm {pipeline_name} not implemented yet")


def load_sam_predictor(sam_name: str) -> SamPredictor:
    if sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {sam_name}")

    name, checkpoint_path = MODEL_MAP[sam_name]
    sam_model = sam_model_registry[name](checkpoint=checkpoint_path)
    sam_model.eval().cuda()
    return SamPredictor(sam_model)
