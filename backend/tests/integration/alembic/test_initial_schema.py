# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import text


def test_database_schema_applied(fxt_session):
    """Test that database tables have been created successfully."""
    result = fxt_session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result.fetchall()]

    assert len(tables) == 8
    assert "alembic_version" in tables
    assert "Project" in tables
    assert "Processor" in tables
    assert "Prompt" in tables
    assert "Sink" in tables
    assert "Source" in tables
    assert "Annotation" in tables
    assert "Label" in tables

    (result,) = fxt_session.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    assert result == "8b19996bfac7"


def test_initial_project_exists(fxt_session):
    """Test that the initial project is created with the default name and active state."""
    result = fxt_session.execute(
        text("SELECT name, active FROM Project")
    ).fetchall()
    assert len(result) == 1
    name, active = result[0]
    assert name == "Project #1"
    assert active == 1  # SQLite uses 1 for True