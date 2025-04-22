from context_learner.pipelines.perdino_pipeline import PerDino
from model_api.models.model import Model
from model_api.models.visual_prompting import SAMLearnableVisualPrompter
from context_learner.pipelines.matcher_pipeline import Matcher
from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.pipelines.pipeline_base import Pipeline
from utils.constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH, MODEL_MAP
from third_party.PersonalizeSAM.per_segment_anything import sam_model_registry
from third_party.PersonalizeSAM.per_segment_anything import SamPredictor
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def load_model(args) -> Pipeline:
    if args.sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {args.sam_name}")

    # Construct backbone
    name, checkpoint_path = MODEL_MAP[args.sam_name]
    if args.sam_name in ["SAM", "MobileSAM"]:
        sam_model = sam_model_registry[name](checkpoint=checkpoint_path).cuda()
        sam_model.eval()
        sam_model = SamPredictor(sam_model)
    elif args.sam_name == "EfficientViT-SAM":
        sam_model = create_efficientvit_sam_model(
            name=name, weight_url=checkpoint_path
        ).cuda()
        sam_model.eval()
        sam_model = EfficientViTSamPredictor(sam_model)
    else:
        raise NotImplementedError(f"Model {args.sam_name} not implemented yet")

    # Construct pipeline
    if args.pipeline == "PerSAMModular":
        return PerSam(sam_model, args)
    elif args.pipeline == "PerDinoModular":
        return PerDino(sam_model, args)
    elif args.pipeline == "MatcherModular":
        return Matcher(sam_model, args)
    elif args.pipeline == "PerSAMMAPI":
        # Only MobileSAM, no backbone choice
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        return SAMLearnableVisualPrompter(encoder, decoder)
    else:
        raise NotImplementedError(f"Algorithm {args.pipeline} not implemented yet")


def load_sam_predictor(sam_name: str) -> SamPredictor:
    if sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {sam_name}")

    name, checkpoint_path = MODEL_MAP[sam_name]
    sam_model = sam_model_registry[name](checkpoint=checkpoint_path)
    sam_model.eval().cuda()
    return SamPredictor(sam_model)
