# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from fastapi import Response, status

from db.repository.common import ResourceNotFoundError
from db.repository.pipeline import PipelineRepository
from dependencies import SessionDep
from routers import pipelines_router

logger = logging.getLogger(__name__)


@pipelines_router.delete(
    path="/{pipeline_id}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Successfully deleted the pipeline.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while deleting the pipeline.",
        },
    },
)
def delete_pipeline(pipeline_id: UUID, db_session: SessionDep) -> Response:
    """
    Delete the specified pipeline.
    """
    logger.debug(f"Received DELETE pipeline {pipeline_id} request.")
    repo = PipelineRepository(db_session)
    try:
        repo.delete_pipeline(pipeline_id)
    except ResourceNotFoundError:
        logger.warning(f"Pipeline with id {pipeline_id} not found during delete operation.")

    logger.info(f"Successfully deleted pipeline with id {pipeline_id}.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
