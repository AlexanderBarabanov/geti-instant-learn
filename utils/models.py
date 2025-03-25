import os
from algorithms import PerDinoPredictor, PerSamPredictor
from model_api.models.model import Model
from model_api.models.visual_prompting import SAMLearnableVisualPrompter
from utils.constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH, MODEL_MAP
from PersonalizeSAM.per_segment_anything import sam_model_registry
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from dinov2.models import vision_transformer
from dinov2.models.vision_transformer import DinoVisionTransformer
import dinov2.utils.utils as dinov2_utils
from PersonalizeSAM.per_segment_anything import SamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def load_model(
    sam_name="SAM", algo_name="PerSAM"
) -> PerSamPredictor | SAMLearnableVisualPrompter | PerDinoPredictor:
    if sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {sam_name}")

    name, checkpoint_path = MODEL_MAP[sam_name]
    if sam_name == "MobileSAM-MAPI":
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        return SAMLearnableVisualPrompter(encoder, decoder)
    elif sam_name in ["SAM", "MobileSAM"]:
        sam_model = sam_model_registry[name](checkpoint=checkpoint_path).cuda()
        sam_model.eval()
        sam_model = SamPredictor(sam_model)
    elif sam_name == "EfficientViT-SAM":
        sam_model = create_efficientvit_sam_model(
            name=name, weight_url=checkpoint_path
        ).cuda()
        sam_model.eval()
        sam_model = EfficientViTSamPredictor(sam_model)
    else:
        raise NotImplementedError(f"Model {sam_name} not implemented yet")

    if algo_name == "PerSAM":
        return PerSamPredictor(sam_model)
    elif algo_name == "PerDINO":
        dino_model = load_dino_model()
        return PerDinoPredictor(sam_model, dino_model)
    else:
        raise NotImplementedError(f"Algorithm {algo_name} not implemented yet")


def load_dino_model() -> DinoVisionTransformer:
    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2: DinoVisionTransformer = vision_transformer.__dict__["vit_large"](
        **dinov2_kwargs
    )
    path = os.path.expanduser("~/data/dinov2_vitl14_pretrain.pth")
    dinov2_utils.load_pretrained_weights(dinov2, path, "teacher")
    dinov2.eval()
    dinov2.to(device="cuda")
    return dinov2


def load_sam_predictor(sam_name: str) -> SamPredictor:
    if sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {sam_name}")

    name, checkpoint_path = MODEL_MAP[sam_name]
    sam_model = sam_model_registry[name](checkpoint=checkpoint_path)
    sam_model.eval().cuda()
    return SamPredictor(sam_model)
