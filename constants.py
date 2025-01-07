import os.path
DATA_PATH = os.path.expanduser(os.path.join("~", "data"))

SAM_MODEL_WEIGHTS = os.path.join(DATA_PATH, "sam_vit_h_4b8939.pth")
MOBILE_SAM_WEIGHTS = os.path.join(DATA_PATH, "data/mobile_sam.pt")
# SAM2_1_MODEL_WEIGHTS = "TODO"
DEFAULT_MODEL = "SAM"
MODEL_MAP = {
    "SAM": ["vit_h", SAM_MODEL_WEIGHTS],
    "MobileSAM": ["vit_t", MOBILE_SAM_WEIGHTS],  # 1024x1024 input resolution
    # "SAM2.1": ["vit_h", SAM2_1_MODEL_WEIGHTS],
    "EfficientViT-SAM": ["efficientvit-sam-l0", "efficientvit/assets/checkpoints/efficientvit_sam_l0.pt"],  # 512x512 input resolution
    "MobileSAM-MAPI": ["",""]
}

DATASETS = ["all", "DAVIS", "PerSeg",]
ALGORITHMS = ["all", "persam", "p2sam"]
MAPI_ENCODER_PATH = os.path.join(DATA_PATH, "otx_models", "sam_vit_b_zsl_encoder", "openvino.xml")
MAPI_DECODER_PATH = os.path.join(DATA_PATH, "otx_models", "sam_vit_b_zsl_decoder", "openvino.xml")
