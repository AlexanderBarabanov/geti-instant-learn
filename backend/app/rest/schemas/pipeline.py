# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field

from rest.schemas.processor import ProcessorSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema


class PipelineSchema(BaseModel):
    id: UUID
    name: str = Field(max_length=80, min_length=1)
    source: SourceSchema | None = None
    processor: ProcessorSchema | None = None
    sink: SinkSchema | None = None


class PipelinePostPayload(BaseModel):
    name: str = Field(max_length=80, min_length=1)


class PipelinePutPayload(BaseModel):
    name: str = Field(max_length=80, min_length=1)


class PipelineListItem(BaseModel):
    id: UUID
    name: str | None = None


class PipelinesListSchema(BaseModel):
    pipelines: list[PipelineListItem]
