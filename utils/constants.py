import os.path

DATA_PATH = os.path.expanduser(os.path.join("~", "data"))
MODEL_MAP = {
    "SAM": ["vit_h", os.path.join(DATA_PATH, "sam_vit_h_4b8939.pth")],
    "MobileSAM": [
        "vit_t",
        os.path.join(DATA_PATH, "mobile_sam.pt"),
    ],  # 1024x1024 input resolution
    "EfficientViT-SAM": [
        "efficientvit-sam-l0",
        os.path.join(DATA_PATH, "efficientvit_sam_l0.pt"),
    ],  # 512x512 input resolution
    "MobileSAM-MAPI": ["", ""],
}

DATASETS = ["all", "PerSeg", "lvis", "lvis_validation"]
PIPELINES = ["all", "MatcherModular", "PerSAMModular", "PerDinoModular", "PerSAMMAPI"]
MAPI_ENCODER_PATH = os.path.join(DATA_PATH, "otx_models", "sam_vit_b_zsl_encoder.xml")
MAPI_DECODER_PATH = os.path.join(DATA_PATH, "otx_models", "sam_vit_b_zsl_decoder.xml")
