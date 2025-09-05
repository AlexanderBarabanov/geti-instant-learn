# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import HTTPException, Response, status

from db.repository.pipeline import PipelineRepository
from dependencies import SessionDep
from rest.schemas.pipeline import PipelineSchema
from rest.schemas.processor import ProcessorSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.get(
    path="/active",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved the configuration of the currently active pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving the active pipeline configuration.",
        },
    },
)
def get_active_pipeline(db_session: SessionDep) -> PipelineSchema | Response:
    """
    Retrieve the configuration of the currently active pipeline.
    """
    logger.debug("Received GET active pipeline request.")
    repo = PipelineRepository(db_session)

    try:
        active_pipeline = repo.get_active_pipeline()
        if not active_pipeline:
            logger.info("No active pipeline found.")
            return Response(status_code=status.HTTP_200_OK, content="No active pipeline.")

        logger.info(f"Active pipeline retrieved: {active_pipeline.name} with id {active_pipeline.id}")

        pipeline_response = PipelineSchema(id=active_pipeline.id, name=active_pipeline.name)
        if active_pipeline.source:
            pipeline_response.source = SourceSchema(
                id=active_pipeline.source.id,
                type=active_pipeline.source.type,
                config=active_pipeline.source.config,
            )

        if active_pipeline.processor:
            pipeline_response.processor = ProcessorSchema(
                id=active_pipeline.processor.id,
                type=active_pipeline.processor.type,
                config=active_pipeline.processor.config,
                name=active_pipeline.processor.name,
            )

        if active_pipeline.sink:
            pipeline_response.sink = SinkSchema(
                id=active_pipeline.sink.id,
                config=active_pipeline.sink.config,
            )

        return pipeline_response

    except Exception as e:
        logger.exception(f"Unexpected error during retrieval of active pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the active pipeline due to internal server error.",
        )
