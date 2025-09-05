# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.model import Base

if TYPE_CHECKING:
    from db.model.pipeline import Pipeline
    from db.model.prompt import Prompt


class Label(Base):
    __tablename__ = "Label"
    name: Mapped[str] = mapped_column(nullable=False)
    color: Mapped[str] = mapped_column(nullable=False)
    pipeline_id: Mapped[UUID | None] = mapped_column(ForeignKey("Pipeline.id", ondelete="CASCADE"))
    prompt_id: Mapped[UUID | None] = mapped_column(ForeignKey("Prompt.id", ondelete="CASCADE"))
    prompt: Mapped["Prompt"] = relationship(back_populates="labels")
    pipeline: Mapped["Pipeline"] = relationship(back_populates="labels")
    __table_args__ = (CheckConstraint("pipeline_id IS NOT NULL OR prompt_id IS NOT NULL", name="label_parent_check"),)
