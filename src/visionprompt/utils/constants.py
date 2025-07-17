# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path


class SAMModelName(Enum):
    """Enum for SAM model types."""

    SAM = "SAM"
    MOBILE_SAM = "MobileSAM"
    EFFICIENT_VIT_SAM = "EfficientViT-SAM"
    SAM_HQ = "SAM-HQ"
    SAM_HQ_TINY = "SAM-HQ-tiny"
    SAM_FAST = "SAM-Fast"


class PipelineName(Enum):
    """Enum for pipeline types."""

    GROUNDING_DINO_SAM = "GroundingDinoSAM"
    MATCHER = "Matcher"
    PER_SAM = "PerSAM"
    PER_DINO = "PerDino"
    PER_SAM_MAPI = "PerSAMMAPI"
    SOFT_MATCHER = "SoftMatcher"


class DatasetName(Enum):
    """Enum for dataset names."""

    PERSEG = "PerSeg"
    LVIS = "lvis"
    LVIS_VALIDATION = "lvis_validation"


DATA_PATH = Path("~/data").expanduser()
MODEL_MAP = {
    SAMModelName.SAM: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha_sum": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    },
    SAMModelName.MOBILE_SAM: {  # 1024x1024 input resolution
        "registry_name": "vit_t",
        "local_filename": "mobile_sam.pt",
        "download_url": "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt",
        "sha_sum": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    },
    SAMModelName.EFFICIENT_VIT_SAM: {  # 512x512 input resolution
        "registry_name": "efficientvit-sam-l0",
        "local_filename": "efficientvit_sam_l0.pt",
        "download_url": "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt",
        "sha_sum": "c4f994b01a16d48bcf2fbbb089448cfbf58fae5811edfa8113c953b8b8cc64b8",
    },
    SAMModelName.SAM_HQ: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_hq_vit_h.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "sha_sum": "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
    },
    SAMModelName.SAM_HQ_TINY: {  # 1024x1024 input resolution
        "registry_name": "vit_tiny",
        "local_filename": "sam_hq_vit_tiny.pth",
        "download_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
        "sha_sum": "0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196",
    },
    SAMModelName.SAM_FAST: {  # 1024x1024 input resolution
        "registry_name": "vit_h",
        "local_filename": "sam_vit_h_4b8939.pth",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha_sum": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    },
}


MAPI_ENCODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_encoder.xml")
MAPI_DECODER_PATH = DATA_PATH.joinpath("otx_models", "sam_vit_b_zsl_decoder.xml")

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
