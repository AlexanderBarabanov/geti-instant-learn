# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.model.processor import ProcessorType
from db.model.source import SourceType
from db.repository.common import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from dependencies import SessionDep  # type: ignore
from rest.endpoints.projects import create as create_mod
from rest.endpoints.projects import delete_project, get_active, get_list, get_project, update_project
from routers import projects_router

PROJECT_ID = uuid4()
PROJECT_ID_STR = str(PROJECT_ID)

SECOND_PROJECT_ID = uuid4()
SECOND_PROJECT_ID_STR = str(SECOND_PROJECT_ID)

SOURCE_ID = uuid4()
PROCESSOR_ID = uuid4()
SINK_ID = uuid4()


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class _FakeProject:
    def __init__(self, project_id: UUID, name: str):
        self.id = project_id
        self.name = name


@pytest.mark.parametrize(
    "behavior,expected_status,expected_location,expected_detail",
    [
        ("success", 201, f"/projects/{PROJECT_ID_STR}", None),
        ("conflict", 409, None, f"Project with id '{PROJECT_ID_STR}' already exists."),
        ("error", 500, None, "Failed to create a project due to internal server error."),
    ],
)
def test_create_project(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    expected_status: int,
    expected_location: str | None,
    expected_detail: str | None,
):
    class FakeRepo:
        def __init__(self, session):
            pass

        def create_project(self, name: str, project_id):
            if behavior == "success":
                assert project_id == PROJECT_ID
                return _FakeProject(PROJECT_ID, name)
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
    response = client.post("/api/v1/projects", json=payload)

    assert response.status_code == expected_status
    if expected_location:
        assert response.headers.get("Location") == expected_location
        assert response.text == ""
    else:
        assert "Location" not in response.headers
    if expected_detail:
        assert response.json()["detail"] == expected_detail
    else:
        assert response.text == ""


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 204, None),
        ("missing", 204, None),
        ("error", 500, f"Failed to delete project with id {PROJECT_ID_STR} due to an internal error."),
    ],
)
def test_delete_project(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    expected_status: int,
    expected_detail: str | None,
):
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

    monkeypatch.setattr(delete_project, "ProjectRepository", FakeRepo)

    resp = client.delete(f"/api/v1/projects/{PROJECT_ID_STR}")

    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    else:
        # FastAPI returns empty body for Response(status_code=204)
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
    class FakeActiveProject:
        def __init__(self, project_id, name):
            self.id = project_id
            self.name = name
            self.source = None
            self.processor = None
            self.sink = None

    class FakeRepo:
        def __init__(self, session):
            pass

        def get_active_project(self):
            if behavior == "success":
                return FakeActiveProject(PROJECT_ID, "activeproj")
            if behavior == "notfound":
                return None
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_active, "ProjectRepository", FakeRepo)

    resp = client.get("/api/v1/projects/active")
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    else:
        data = resp.json()
        assert data["id"] == PROJECT_ID_STR
        assert data["name"] == "activeproj"
        assert data["source"] is None
        assert data["processor"] is None
        assert data["sink"] is None


@pytest.mark.parametrize(
    "behavior,expected_status,expected_count,expected_detail",
    [
        ("empty", 200, 0, None),
        ("some", 200, 2, None),
        ("error", 500, None, "Failed to retrieve projects due to internal server error."),
    ],
)
def test_get_projects_list(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                           behavior: str, expected_status: int,
                           expected_count: int | None, expected_detail: str | None):
    class FakeRepo:
        def __init__(self, session):
            pass

        def get_all_projects(self):
            if behavior == "empty":
                return []
            if behavior == "some":
                return [
                    _FakeProject(PROJECT_ID, "proj1"),
                    _FakeProject(SECOND_PROJECT_ID, "proj2"),
                ]
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_list, "ProjectRepository", FakeRepo)

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
        assert {p["id"] for p in projects} == {PROJECT_ID_STR, SECOND_PROJECT_ID_STR}
        # Optional: verify names align with IDs
        id_to_name = {p["id"]: p["name"] for p in projects}
        assert id_to_name[PROJECT_ID_STR] == "proj1"
        assert id_to_name[SECOND_PROJECT_ID_STR] == "proj2"


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("minimal", 200, None),
        ("full", 200, None),
        ("notfound", 404, f"Project with id {PROJECT_ID_STR} not found."),
        ("error", 500, "Failed to retrieve the project due to internal server error."),
    ],
)
def test_get_project(client: TestClient,
                     monkeypatch: pytest.MonkeyPatch,
                     behavior: str,
                     expected_status: int,
                     expected_detail: str | None):
    class _FakeSource:
        def __init__(self):
            self.id = SOURCE_ID
            self.type = SourceType.VIDEO_FILE
            self.config = {"path": "/tmp/data.txt"}

    class _FakeProcessor:
        def __init__(self):
            self.id = PROCESSOR_ID
            self.type = ProcessorType.DUMMY
            self.config = {"mode": "fast"}
            self.name = "proc1"

    class _FakeSink:
        def __init__(self):
            self.id = SINK_ID
            self.config = {"dest": "stdout"}

    class FakeProjectFull:
        def __init__(self):
            self.id = PROJECT_ID
            self.name = "fullproj"
            self.source = _FakeSource()
            self.processor = _FakeProcessor()
            self.sink = _FakeSink()

    class FakeProjectMinimal:
        def __init__(self):
            self.id = PROJECT_ID
            self.name = "minproj"
            self.source = None
            self.processor = None
            self.sink = None

    class FakeRepo:
        def __init__(self, session):
            pass

        def get_project_by_id(self, project_id: UUID):
            assert project_id == PROJECT_ID
            if behavior == "minimal":
                return FakeProjectMinimal()
            if behavior == "full":
                return FakeProjectFull()
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(get_project, "ProjectRepository", FakeRepo)

    resp = client.get(f"/api/v1/projects/{PROJECT_ID_STR}")
    assert resp.status_code == expected_status

    if expected_detail:
        assert resp.json()["detail"] == expected_detail
        return

    data = resp.json()
    assert data["id"] == PROJECT_ID_STR
    if behavior == "minimal":
        assert data["name"] == "minproj"
        assert data["source"] is None
        assert data["processor"] is None
        assert data["sink"] is None
    if behavior == "full":
        assert data["name"] == "fullproj"
        assert data["source"] == {
            "id": str(SOURCE_ID),
            "type": "VIDEO_FILE",  # UPDATED expected value
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


@pytest.mark.parametrize(
    "behavior,expected_status,expected_detail",
    [
        ("success", 200, None),
        ("notfound", 404, f"Project with ID {PROJECT_ID_STR} not found."),
        ("error", 500, "Failed to update the project due to internal server error."),
    ],
)
def test_update_project(client: TestClient,
                        monkeypatch: pytest.MonkeyPatch,
                        behavior: str,
                        expected_status: int,
                        expected_detail: str | None):
    NEW_NAME = "renamed"

    class _UpdatedProject:
        def __init__(self, project_id: UUID, name: str):
            self.id = project_id
            self.name = name
            # Simulate no related components
            self.source = None
            self.processor = None
            self.sink = None

    class FakeRepo:
        def __init__(self, session):
            pass

        def update_project(self, project_id: UUID, new_name: str):
            assert project_id == PROJECT_ID
            assert new_name == NEW_NAME
            if behavior == "success":
                return _UpdatedProject(PROJECT_ID, new_name)
            if behavior == "notfound":
                raise ResourceNotFoundError(ResourceType.PROJECT, resource_id=str(project_id))
            if behavior == "error":
                raise RuntimeError("boom")
            raise AssertionError("Unhandled behavior")

    monkeypatch.setattr(update_project, "ProjectRepository", FakeRepo)

    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": NEW_NAME})
    assert resp.status_code == expected_status
    if expected_detail:
        assert resp.json()["detail"] == expected_detail
    else:
        data = resp.json()
        assert data["id"] == PROJECT_ID_STR
        assert data["name"] == NEW_NAME
        assert data["source"] is None
        assert data["processor"] is None
        assert data["sink"] is None


def test_update_project_validation_error(client: TestClient):
    # Empty name violates min_length=1 -> 422
    resp = client.put(f"/api/v1/projects/{PROJECT_ID_STR}", json={"name": ""})
    assert resp.status_code == 422