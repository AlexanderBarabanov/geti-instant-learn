# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from db.repository.pipeline import Pipeline, PipelineRepository
from dependencies import SessionDep
from rest.schemas.pipeline import PipelineListItem, PipelinesListSchema
from routers import pipelines_router

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@pipelines_router.get(
    path="",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved a list of all available pipeline configurations.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving available pipeline configurations.",
        },
    },
)
def get_pipelines_list(db_session: SessionDep) -> PipelinesListSchema:
    """
    Retrieve a list of all available pipeline configurations.
    """
    logger.debug("Received GET pipelines request.")
    repo = PipelineRepository(db_session)

    try:
        pipelines: Sequence[Pipeline] = repo.get_all_pipelines()
        pipeline_items = [PipelineListItem(id=pipeline.id, name=pipeline.name) for pipeline in pipelines]
        logger.debug(f"Retrieved {len(pipeline_items)} pipelines: {pipeline_items}")

        return PipelinesListSchema(pipelines=pipeline_items)

    except Exception:
        logger.exception("Unexpected error during retrieving pipelines list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pipelines due to internal server error.",
        )
