# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import HTTPException, status

from db.repository.common import ResourceNotFoundError
from db.repository.pipeline import PipelineRepository
from dependencies import SessionDep
from rest.schemas.pipeline import PipelinePutPayload, PipelineSchema
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.put(
    path="/{pipeline_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully updates the configuration for the pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while updating the configuration of the pipeline.",
        },
    },
)
def update_pipeline(pipeline_id: UUID, payload: PipelinePutPayload, db_session: SessionDep) -> PipelineSchema:
    """
    Update the pipeline's configuration.
    """
    logger.debug(f"Received PUT pipeline {pipeline_id} request.")
    repo = PipelineRepository(db_session)
    try:
        updated_pipeline = repo.update_pipeline(pipeline_id=pipeline_id, new_name=payload.name)
        logger.info(f"Successfully updated pipeline with id {updated_pipeline.id}, new name: {updated_pipeline.name}")

        return PipelineSchema(id=updated_pipeline.id, name=updated_pipeline.name)

    except ResourceNotFoundError as e:
        logger.exception(f"Pipeline update failed: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during pipeline update: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update the pipeline due to internal server error.",
        )
