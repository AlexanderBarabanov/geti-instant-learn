# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.model.processor import ProcessorType
from db.model.source import SourceType
from db.repository.common import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from dependencies import SessionDep  # type: ignore
from rest.endpoints.projects import create as create_mod
from rest.endpoints.projects import delete_project as delete_mod
from rest.endpoints.projects import get_active as get_active_mod
from rest.endpoints.projects import get_list as get_list_mod
from rest.endpoints.projects import get_project as get_project_mod
from rest.endpoints.projects import update_project as update_project_mod
from routers import projects_router

# Reusable IDs
PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)
SECOND_PROJECT_ID = uuid4()
SECOND_PROJECT_ID_STR = str(SECOND_PROJECT_ID)
SOURCE_ID = uuid4()
PROCESSOR_ID = uuid4()
SINK_ID = uuid4()


# Fake model dataclasses
@dataclass
class FakeSource:
    id: UUID
    type: SourceType
    config: dict


@dataclass
class FakeProcessor:
    id: UUID
    type: ProcessorType
    config: dict
    name: str


@dataclass
class FakeSink:
    id: UUID
    config: dict


@dataclass
class FakeProject:
    id: UUID
    name: str
    source: FakeSource | None = None
    processor: FakeProcessor | None = None
    sink: FakeSink | None = None


# Factories
def make_source() -> FakeSource:
    return FakeSource(
        id=SOURCE_ID,
        type=SourceType.VIDEO_FILE,
        config={"path": "/tmp/data.txt"},
    )


def make_processor() -> FakeProcessor:
    return FakeProcessor(
        id=PROCESSOR_ID,
        type=ProcessorType.DUMMY,
        config={"mode": "fast"},
        name="proc1",
    )


def make_sink() -> FakeSink:
    return FakeSink(
        id=SINK_ID,
        config={"dest": "stdout"},
    )


def make_project(
    project_id: UUID,
    name: str,
    source: FakeSource | None = None,
    processor: FakeProcessor | None = None,
    sink: FakeSink | None = None,
) -> FakeProject:
    return FakeProject(
        id=project_id,
        name=name,
        source=source,
        processor=processor,
        sink=sink,
    )


# Assertion helpers
def assert_minimal_project_payload(data: dict, project_id: str, name: str):
    assert data["id"] == project_id
    assert data["name"] == name
    assert data["source"] is None
    assert data["processor"] is None
    assert data["sink"] is None


def assert_full_project_payload(data: dict):
    assert data["id"] == PROJECT_ID_STR
    assert data["name"] == "fullproj"
    assert data["source"] == {
        "id": str(SOURCE_ID),
        "type": "VIDEO_FILE",
        "config": {"path": "/tmp/data.txt"},
    }
    assert data["processor"] == {
        "id": str(PROCESSOR_ID),
        "type": "DUMMY",
        "config": {"mode": "fast"},
        "name": "proc1",
    }
    assert data["sink"] == {
        "id": str(SINK_ID),
        "config": {"dest": "stdout"},
    }


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.parametrize(
    "behavior,expected_status,expected_location,expected_detail",
    [
        ("success", 201, f"/projects/{PROJECT_ID_STR}", None),
        ("conflict", 409, None, f"Project with id '{PROJECT_ID_STR}' already exists."),
        ("error", 500, None, "Failed to create a project due to internal server error."),
    ],
)
def test_create_project(client, monkeypatch, behavior, expected_status, expected_location, expected_detail):
    class FakeRepo:
        def __init__(self, session):
            pass

        def create_project(self, name: str, project_id):
            if behavior == "success":
                assert project_id == PROJECT_ID
                return make_project(PROJECT_ID, name)
            if behavior == "conflict":
                raise ResourceAlreadyExistsError(
                    ResourceType.PROJECT,
                    str(project_id),
                    raised_by="id",
                )
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(create_mod, "ProjectRepository", FakeRepo)

    payload = {"id": PROJECT_ID_STR, "name": "myproj"}
    resp = client.post("/api/v1/projects", json=payload)

    assert resp.status_code == expected_status
    if expected_location:
        assert resp.headers.get("Location") == expected_location
        assert resp.text == ""
    else:
        assert "Location" not in resp.headers
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    else:
        assert resp.text == ""


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 204, None),
        ("missing", 204, None),
        ("error", 500, f"Failed to delete project with id {PROJECT_ID_STR} due to an internal error."),
    ],
)
def test_delete_project(client, monkeypatch, behavior, expected_status, expected_detail):
    class FakeRepo:
        def __init__(self, session):
            pass

        def delete_project(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "success":
                return None
            if behavior == "missing":
                raise ResourceNotFoundError(ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(delete_mod, "ProjectRepository", FakeRepo)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    else:
        assert resp.text == ""


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 200, None),
        ("notfound", 404, "No active project found."),
        ("error", 500, "Failed to retrieve the active project due to internal server error."),
    ],
)
def test_get_active_project(client, monkeypatch, behavior, expected_status, expected_detail):
    class FakeRepo:
        def __init__(self, session):
            pass

        def get_active_project(self):
            if behavior == "success":
                return make_project(PROJECT_ID, "activeproj")
            if behavior == "notfound":
                return None
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_active_mod, "ProjectRepository", FakeRepo)

    resp = client.get("/api/v1/projects/active")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    elif behavior == "success":
        assert_minimal_project_payload(resp.json(), PROJECT_ID_STR, "activeproj")


@pytest.mark.parametrize(
    "behavior,expected_status,expected_count,expected_detail",
    [
        ("empty", 200, 0, None),
        ("some", 200, 2, None),
        ("error", 500, None, "Failed to retrieve projects due to internal server error."),
    ],
)
def test_get_projects_list(client, monkeypatch, behavior, expected_status, expected_count, expected_detail):
    class FakeRepo:
        def __init__(self, session):
            pass

        def get_all_projects(self):
            if behavior == "empty":
                return []
            if behavior == "some":
                return [
                    make_project(PROJECT_ID, "proj1"),
                    make_project(SECOND_PROJECT_ID, "proj2"),
                ]
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_list_mod, "ProjectRepository", FakeRepo)

    resp = client.get("/api/v1/projects")
    assert resp.status_code == expected_status

    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return

    data = resp.json()
    assert "projects" in data
    projects = data["projects"]
    assert len(projects) == expected_count
    if behavior == "some":
        ids = {p["id"] for p in projects}
        assert ids == {PROJECT_ID_STR, SECOND_PROJECT_ID_STR}
        lookup = {p["id"]: p["name"] for p in projects}
        assert lookup[PROJECT_ID_STR] == "proj1"
        assert lookup[SECOND_PROJECT_ID_STR] == "proj2"


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("minimal", 200, None),
        ("full", 200, None),
        ("notfound", 404, f"Project with id {PROJECT_ID_STR} not found."),
        ("error", 500, "Failed to retrieve the project due to internal server error."),
    ],
)
def test_get_project(client, monkeypatch, behavior, expected_status, expected_detail):
    class FakeRepo:
        def __init__(self, session):
            pass

        def get_project_by_id(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "minimal":
                return make_project(PROJECT_ID, "minproj")
            if behavior == "full":
                return make_project(
                    PROJECT_ID,
                    "fullproj",
                    source=make_source(),
                    processor=make_processor(),
                    sink=make_sink(),
                )
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_project_mod, "ProjectRepository", FakeRepo)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status

    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return

    data = resp.json()
    if behavior == "minimal":
        assert_minimal_project_payload(data, PROJECT_ID_STR, "minproj")
    if behavior == "full":
        assert_full_project_payload(data)


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 200, None),
        ("notfound", 404, f"Project with ID {PROJECT_ID_STR} not found."),
        ("error", 500, "Failed to update the project due to internal server error."),
    ],
)
def test_update_project(client, monkeypatch, behavior, expected_status, expected_detail):
    NEW_NAME = "renamed"

    class FakeRepo:
        def __init__(self, session):
            pass

        def update_project(self, project_id: UUID, new_name: str):
            assert project_id == PROJECT_ID
            assert new_name == NEW_NAME
            if behavior == "success":
                return make_project(PROJECT_ID, NEW_NAME)
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(update_project_mod, "ProjectRepository", FakeRepo)

    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": NEW_NAME})
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    elif behavior == "success":
        assert_minimal_project_payload(resp.json(), PROJECT_ID_STR, NEW_NAME)


def test_update_project_validation_error(client):
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 422
