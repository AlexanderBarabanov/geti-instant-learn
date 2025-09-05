# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.model import Pipeline
from db.repository.common import BaseRepository, ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType

logger = logging.getLogger(__name__)


class PipelineRepository(BaseRepository):
    def __init__(self, session: Session):
        self.session = session

    def create_pipeline(self, name: str) -> Pipeline:
        """
        Create a new Pipeline in the database.

        Raises:
            ValueError: If a pipeline with the given name already exists.
        """
        existing_pipeline: Pipeline | None = self.session.scalars(select(Pipeline).where(Pipeline.name == name)).first()
        if existing_pipeline:
            raise ResourceAlreadyExistsError(resource_type=ResourceType.PIPELINE, resource_name=name)

        new_pipeline = Pipeline(name=name)
        self.session.add(new_pipeline)
        self.session.flush()
        self.session.refresh(new_pipeline)
        self.set_active_pipeline(pipeline_id=new_pipeline.id)
        self.session.commit()
        return new_pipeline

    def get_active_pipeline(self) -> Pipeline | None:
        """
        Retrieve the currently active pipeline.

        Returns:
            The active Pipeline instance, or None if no pipeline is active.

        Raises:
            RuntimeError: If more than one active pipeline is found.
        """
        active_pipelines = self.session.scalars(select(Pipeline).where(Pipeline.active)).all()
        if len(active_pipelines) == 0:
            return None
        if len(active_pipelines) > 1:
            raise RuntimeError("More than one active pipeline found.")
        return active_pipelines[0]

    def set_active_pipeline(self, pipeline_id: UUID) -> None:
        """
        Set the pipeline with the given ID as active, deactivating any currently active pipeline.

        Args:
            pipeline_id: The UUID of the pipeline to activate.
        """
        active_pipeline: Pipeline | None = self.get_active_pipeline()
        if active_pipeline:
            active_pipeline.active = False

        pipeline_to_activate: Pipeline | None = self.session.scalars(
            select(Pipeline).where(Pipeline.id == pipeline_id)
        ).first()
        if pipeline_to_activate:
            pipeline_to_activate.active = True
            self.session.commit()
        else:
            self.session.rollback()
            raise ValueError(f"Pipeline with id {pipeline_id} not found.")

    def update_pipeline(self, pipeline_id: UUID, new_name: str) -> Pipeline:
        """
        Update the pipeline with the given ID with the provided info.

        Args:
            pipeline_id: The UUID of the pipeline to update.
            new_name: The new name for the pipeline.

        Returns: The updated Pipeline instance, or None if no pipeline with the given ID exists.
        """
        pipeline: Pipeline | None = self.session.scalars(select(Pipeline).where(Pipeline.id == pipeline_id)).first()
        if not pipeline:
            raise ResourceNotFoundError(resource_type=ResourceType.PIPELINE, resource_id=str(pipeline_id))
        pipeline.name = new_name
        self.session.commit()
        self.session.refresh(pipeline)
        return pipeline

    def get_pipeline_by_id(self, pipeline_id: UUID) -> Pipeline:
        """
        Retrieve a pipeline by its ID.

        Returns:
            The Pipeline instance, or None if not found.
        """
        pipeline: Pipeline | None = self.session.scalars(select(Pipeline).where(Pipeline.id == pipeline_id)).first()
        if not pipeline:
            raise ResourceNotFoundError(resource_type=ResourceType.PIPELINE, resource_id=str(pipeline_id))
        return pipeline

    def get_all_pipelines(self) -> Sequence[Pipeline]:
        """
        Retrieve all existing pipelines.

        Returns:
            A list of all Pipeline instances.
        """
        return self.session.scalars(select(Pipeline)).all()

    def delete_pipeline(self, pipeline_id: UUID) -> None:
        """
        Delete the pipeline with the given ID and all related sources, processors, sinks, prompts, and labels.

        Args:
            pipeline_id: The UUID of the pipeline to delete.

        Raises:
            ValueError: If no pipeline with the given ID is found.
        """
        pipeline: Pipeline | None = self.session.scalars(select(Pipeline).where(Pipeline.id == pipeline_id)).first()
        if not pipeline:
            raise ResourceNotFoundError(resource_type=ResourceType.PIPELINE, resource_id=str(pipeline_id))

        self.session.delete(pipeline.source)
        self.session.delete(pipeline.processor)
        self.session.delete(pipeline.sink)

        for prompt in pipeline.prompts:
            self.session.delete(prompt)
        for label in pipeline.labels:
            self.session.delete(label)

        self.session.delete(pipeline)
        self.session.commit()
