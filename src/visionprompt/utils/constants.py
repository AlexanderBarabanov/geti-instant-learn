# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DATA_PATH = Path("~/data").expanduser()
MODEL_MAP = {
    "SAM": {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "MobileSAM": {  # 1024x1024 input resolution
        "registry_name": "vit_t",
        "local_filename": "mobile_sam.pt",
        "download_url": "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt",
    },
    "EfficientViT-SAM": {  # 512x512 input resolution
        "registry_name": "efficientvit-sam-l0",
        "local_filename": "efficientvit_sam_l0.pt",
        "download_url": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt",
    },
    "SAM-HQ": {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_hq_vit_h.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
    },
    "SAM-HQ-tiny": {  # 1024x1024 input resolution
        "registry_name": "vit_tiny",
        "local_filename": "sam_hq_vit_tiny.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
    },
    "SAM-Fast": {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
}


DATASETS = ["all", "PerSeg", "lvis", "lvis_validation"]
PIPELINES = [
    "all",
    "MatcherModular",
    "PerSAMModular",
    "PerDinoModular",
    "PerSAMMAPIModular",
    "SoftMatcherModular",
    "SoftMatcherRFFModular",
    "SoftMatcherBiDirectionalModular",
    "SoftMatcherRFFBiDirectionalModular",
    "SoftMatcherSamplingModular",
    "SoftMatcherRFFSamplingModular",
    "SoftMatcherBiDirectionalSamplingModular",
    "SoftMatcherRFFBiDirectionalSamplingModular",
    "GroundingDinoSAM",
]
MAPI_ENCODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_encoder.xml")
MAPI_DECODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_decoder.xml")
