# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, Response, status

from db.repository.common import ResourceNotFoundError
from db.repository.pipeline import PipelineRepository
from dependencies import SessionDep
from rest.schemas.pipeline import PipelineSchema
from rest.schemas.processor import ProcessorSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.get(
    path="/{pipeline_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration for a pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the configuration of a pipeline.",
        },
    },
)
def get_pipeline(pipeline_id: UUID, db_session: SessionDep) -> PipelineSchema | Response:
    """
    Retrieve the pipeline's configuration.
    """
    logger.debug(f"Received GET pipeline {pipeline_id} request.")

    repo = PipelineRepository(db_session)

    try:
        pipeline = repo.get_pipeline_by_id(pipeline_id)
        pipeline_response = PipelineSchema(id=pipeline.id, name=pipeline.name)
        if pipeline.source:
            pipeline_response.source = SourceSchema(
                id=pipeline.source.id,
                type=pipeline.source.type,
                config=pipeline.source.config,
            )

        if pipeline.processor:
            pipeline_response.processor = ProcessorSchema(
                id=pipeline.processor.id,
                type=pipeline.processor.type,
                config=pipeline.processor.config,
                name=pipeline.processor.name,
            )

        if pipeline.sink:
            pipeline_response.sink = SinkSchema(
                id=pipeline.sink.id,
                config=pipeline.sink.config,
            )

        return pipeline_response

    except ResourceNotFoundError:
        logger.exception(f"Pipeline with id {pipeline_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline with id {pipeline_id} not found.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error during retrieval of a pipeline with id {pipeline_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the pipeline due to internal server error.",
        )
