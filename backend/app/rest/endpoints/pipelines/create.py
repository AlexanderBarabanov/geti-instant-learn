# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import HTTPException, Response, status

from db.repository.common import ResourceAlreadyExistsError
from db.repository.pipeline import PipelineRepository
from dependencies import SessionDep
from rest.schemas.pipeline import PipelinePostPayload
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.post(
    path="",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Successfully created a new pipeline.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Pipeline with this name already exists.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while creating a new pipeline.",
        },
    },
)
def create_pipeline(payload: PipelinePostPayload, db_session: SessionDep) -> Response:
    """Create a new pipeline with the given name."""

    logger.debug(f"Attempting to create pipeline with name: {payload.name}")
    repo = PipelineRepository(db_session)

    try:
        pipeline = repo.create_pipeline(payload.name)
        logger.info(f"Successfully created {pipeline.name} pipeline with id {pipeline.id}")
    except ResourceAlreadyExistsError as e:
        logger.error(f"Pipeline creation failed: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during pipeline creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a pipeline due to internal server error.",
        )

    location = f"/pipelines/{pipeline.id}"
    return Response(status_code=status.HTTP_201_CREATED, headers={"Location": location})
