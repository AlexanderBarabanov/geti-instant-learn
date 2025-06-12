# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from logging import getLogger

import numpy as np
import torch
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from segment_anything_fast.predictor import SamPredictor as SamFastPredictor
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from visionprompt.third_party.dinov2.models.vision_transformer import DinoVisionTransformer
from visionprompt.third_party.PersonalizeSAM.per_segment_anything import (
    SamPredictor,
)
from visionprompt.third_party.PersonalizeSAM.per_segment_anything.modeling.tiny_vit_sam import (
    Attention,
    TinyViT,
)

logger = getLogger("Vision Prompt")


def optimize_dino_model(
    model: DinoVisionTransformer,
    input_image_size: int,
    precision: torch.dtype,
    compile_models: bool,
    verbose: bool = True,
) -> DinoVisionTransformer:
    """This method optimizes the DINO model.

    Args:
        model: The DINO model to optimize.
        input_image_size: The size of the input image.
        precision: The precision to use for the model.
        compile_models: Whether to compile the model.
        verbose: Whether to show the inference time.

    Returns:
        The optimized DINO model.
    """

    def show_inference_time(model: DinoVisionTransformer) -> None:
        random_input = torch.randn(1, 3, input_image_size, input_image_size).to(model.cls_token.dtype).cuda()
        start = time.time()
        _ = model(random_input)
        end = time.time()
        model_size = torch.cuda.memory_allocated() / 1e6
        logger.debug(
            f"Inference time: {end - start:.2f} seconds, FPS: {1 / (end - start):.2f}, "
            f"Memory allocated: {model_size:.2f} MB"
        )

    if verbose:
        logger.debug("DINO model initial inference:")
    show_inference_time(model)
    if precision != torch.float32:
        model = model.to(dtype=precision)
    if verbose:
        logger.debug("Quantized inference:")
        show_inference_time(model)

    if compile_models:
        logger.debug("Compiling model, this can take a while...")
        model = torch.compile(model)
        model(torch.randn(1, 3, input_image_size, input_image_size).to(precision).cuda())
        if verbose:
            logger.debug("Compiled model inference:")
            show_inference_time(model)

    return model


def optimize_sam_model(
    sam_predictor: SamPredictor | SamHQPredictor | SamFastPredictor | EfficientViTSamPredictor,
    precision: torch.dtype,
    compile_models: bool,
    verbose: bool,
) -> SamPredictor | SamHQPredictor | SamFastPredictor | EfficientViTSamPredictor:
    """Optimize a SAM model for inference.

    Optimization is performed by quantizing the model to the specified precision and then compiling the model.
    The model functions are monkey patched to use the correct precision.

    Args:
        sam_predictor: The SAM predictor to optimize.
        precision: The precision to use for the model.
        compile_models: Whether to compile the model.
        verbose: Whether to show detailed optimization logs.

    Returns:
        The optimized SAM model.
    """
    if isinstance(sam_predictor, SamFastPredictor):
        logger.debug("First inference with SamFastPredictor can take while to warm up the model")
        sam_predictor.set_image(np.ones(shape=(1024, 1024, 3), dtype=np.uint8))
        logger.debug("SamFastPredictor model warmed up")
        return sam_predictor
    logger.debug(
        f"Optimizing SAM model for {precision!s} inference "
        f"{'and compiling for current hardware.' if compile_models else ''}",
    )

    def test_inference(sam_predictor: SamPredictor | SamHQPredictor | EfficientViTSamPredictor) -> float:
        start = time.time()
        sam_predictor.set_image(np.ones(shape=(1024, 1024, 3), dtype=np.uint8))
        end = time.time()
        model_size = torch.cuda.memory_allocated() / 1e6
        logger.debug(
            f"Inference time: {end - start:.2f} seconds, FPS: {1 / (end - start):.2f}, "
            f"Memory allocated: {model_size:.2f} MB"
        )
        return end - start

    if verbose:
        logger.debug("SAM model initial inference:")
        intial_inference_duration = test_inference(sam_predictor)

    # Convert all parameters to the specified dtype
    if precision != torch.float32:
        sam_predictor.model.to(precision)
    _monkey_patch_dtype(sam_predictor)

    if verbose:
        logger.debug(f"Quantized {precision!s} inference:")
        optimized_inference_duration = test_inference(sam_predictor)
        logger.debug(f"Quantization speedup: {intial_inference_duration / optimized_inference_duration:.2f}x")

    # Compiling the model grealy improves the inference time
    if compile_models:
        logger.debug("Compiling model, this can take a while...")
        sam_predictor.model.image_encoder = torch.compile(sam_predictor.model.image_encoder)
        sam_predictor.set_image(np.ones(shape=(1024, 1024, 3), dtype=np.uint8))
        if verbose:
            logger.debug("Compiled model inference:")
            optimized_inference_duration = test_inference(sam_predictor)
            logger.debug(
                f"Quantization + Compilation speedup: {intial_inference_duration / optimized_inference_duration:.2f}x"
            )
    logger.debug("Done optimizing SAM model")

    return sam_predictor


def _monkey_patch_preprocess(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the preprocess method to use the correct dtype."""
    original_preprocess = predictor.model.preprocess

    def preprocess_dtype_wrapper(input_tensor: torch.Tensor) -> torch.Tensor:
        output_from_original_preprocess = original_preprocess(input_tensor)
        return output_from_original_preprocess.to(dtype)

    predictor.model.preprocess = preprocess_dtype_wrapper


def _monkey_patch_prompt_encoder(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the prompt encoder methods to use the correct dtype."""
    original_pe_encoding = predictor.model.prompt_encoder.pe_layer._pe_encoding  # noqa: SLF001

    def pe_encoding_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                processed_args.append(arg.to(dtype))
            else:
                processed_args.append(arg)
        args = tuple(processed_args)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float:
                kwargs[key] = value.to(dtype)

        return original_pe_encoding(*args, **kwargs)

    predictor.model.prompt_encoder.pe_layer._pe_encoding = pe_encoding_dtype_wrapper  # noqa: SLF001

    original_prompt_encoder_forward = predictor.model.prompt_encoder.forward

    def prompt_encoder_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        outputs = original_prompt_encoder_forward(*args, **kwargs)
        return [output.to(dtype) for output in outputs]

    predictor.model.prompt_encoder.forward = prompt_encoder_dtype_wrapper


def _monkey_patch_predict_torch(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the predict_torch method to use the correct dtype."""
    original_predict_torch = predictor.predict_torch

    def predict_torch_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                processed_args.append(arg.to(dtype))
            else:
                processed_args.append(arg)
        args = tuple(processed_args)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float:
                kwargs[key] = value.to(dtype)
        outputs = original_predict_torch(*args, **kwargs)
        # SamPredictor.predict internally converts the outputs of predict_torch to numpy, which requires floats
        processed_outputs = []
        for output in outputs:
            if isinstance(output, torch.Tensor) and output.dtype == dtype:
                processed_outputs.append(output.to(torch.float))
            else:
                processed_outputs.append(output)
        return processed_outputs

    predictor.predict_torch = predict_torch_dtype_wrapper


def _monkey_patch_tinyvit_architecture(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Change model.forward to use x.to_dtype before calling the original forward function.

    The 'ab' attribute of the Attention layers are not correctly set to the correct dtype.
    """
    if not isinstance(predictor.model.image_encoder, TinyViT):
        return
    original_forward = predictor.model.image_encoder.forward

    def forward_dtype_wrapper(self_tinyvit, x_input_to_tinyvit, *args_tinyvit, **kwargs_tinyvit) -> torch.Tensor:
        x_input_to_tinyvit = x_input_to_tinyvit.to(dtype)

        # The 'ab' attribute of the Attention layers are not correctly set to the correct dtype.
        if not self_tinyvit.training:
            for module in self_tinyvit.modules():
                if (
                    isinstance(module, Attention)
                    and hasattr(module, "ab")
                    and module.ab is not None
                    and module.ab.dtype != dtype
                ):
                    module.ab = module.ab.to(dtype)

        return original_forward(x_input_to_tinyvit, *args_tinyvit, **kwargs_tinyvit)

    predictor.model.image_encoder.forward = forward_dtype_wrapper.__get__(
        predictor.model.image_encoder, predictor.model.image_encoder.__class__
    )


def _monkey_patch_dtype(predictor: SamPredictor | SamHQPredictor) -> None:
    """Monkey patch the predictor to use the correct dtype for the model.

    The input to the model has to be transformed to the correct dtype before being passed to the model.

    Args:
        predictor: The predictor to monkey patch.
    """
    # EfficientViTSamPredictor already dynamically handles the dtype of the input
    if isinstance(predictor, EfficientViTSamPredictor):
        return

    dtype = predictor.model.mask_decoder.iou_prediction_head.layers[0].weight.dtype

    _monkey_patch_preprocess(predictor, dtype)
    _monkey_patch_prompt_encoder(predictor, dtype)
    _monkey_patch_predict_torch(predictor, dtype)
    _monkey_patch_tinyvit_architecture(predictor, dtype)
