"""Preprocessors."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.processes.preprocessors.resize_images import ResizeImages
from getiprompt.processes.preprocessors.resize_masks import ResizeMasks

__all__ = ["ResizeImages", "ResizeMasks"]
