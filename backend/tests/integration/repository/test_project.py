# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest

from db.model import Project
from db.repository.common import ResourceAlreadyExistsError, ResourceNotFoundError
from db.repository.project import ProjectRepository


@pytest.mark.parametrize("project_id", [None, uuid4()])
def test_create_project(fxt_session, request, fxt_clean_table, project_id) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(session=fxt_session)
    name = "test_project"

    new_project = repo.create_project(name=name, project_id=project_id)

    assert new_project.name == name
    if project_id is None:
        assert isinstance(new_project.id, UUID)
    else:
        assert new_project.id == project_id
    assert new_project.active is True


def test_create_duplicate_name_exception(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    repo.create_project(name="alpha")
    with pytest.raises(ResourceAlreadyExistsError) as exc:
        repo.create_project(name="alpha")
    assert "name" in str(exc.value)


def test_create_duplicate_id_exception(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    pid = uuid4()
    repo.create_project(name="p1", project_id=pid)
    with pytest.raises(ResourceAlreadyExistsError) as exc:
        repo.create_project(name="p2", project_id=pid)
    assert "id" in str(exc.value)


def test_get_active_project(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    created = repo.create_project(name="active_one")
    active = repo.get_active_project()
    assert active.id == created.id
    assert active.active is True


def test_get_active_project_none(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    assert repo.get_active_project() is None


def test_get_active_project_multiple_raises(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    p1 = repo.create_project(name="p1")
    p2 = repo.create_project(name="p2")
    # force second active without deactivating first (simulate data corruption)
    p1.active = True
    p2.active = True
    fxt_session.commit()

    with pytest.raises(RuntimeError):
        repo.get_active_project()


def test_set_active_project_switch(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    p1 = repo.create_project(name="first")
    p2 = repo.create_project(name="second")

    # ensure second became active due to creation
    assert repo.get_active_project().id == p2.id

    repo.set_active_project(project_id=p1.id)
    assert repo.get_active_project().id == p1.id
    fxt_session.refresh(p2)
    assert p2.active is False


def test_set_active_project_not_found(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    with pytest.raises(ResourceNotFoundError):
        repo.set_active_project(project_id=uuid4())


def test_update_project_success(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    p = repo.create_project(name="old")
    updated = repo.update_project(project_id=p.id, new_name="new")
    assert updated.name == "new"
    fetched = repo.get_project_by_id(p.id)
    assert fetched.name == "new"


def test_update_project_not_found(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    with pytest.raises(ResourceNotFoundError):
        repo.update_project(project_id=uuid4(), new_name="x")


def test_get_project_by_id_success(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    p = repo.create_project(name="one")
    got = repo.get_project_by_id(p.id)
    assert got.id == p.id


def test_get_project_by_id_not_found(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    with pytest.raises(ResourceNotFoundError):
        repo.get_project_by_id(uuid4())


def test_get_all_projects(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    created = {repo.create_project(name=f"p{i}").id for i in range(3)}
    all_projects = repo.get_all_projects()
    assert len(all_projects) == 3
    assert {p.id for p in all_projects} == created


def test_delete_project_success(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    p = repo.create_project(name="todelete")
    repo.delete_project(project_id=p.id)
    with pytest.raises(ResourceNotFoundError):
        repo.get_project_by_id(p.id)


def test_delete_project_not_found(fxt_session, request, fxt_clean_table) -> None:
    request.addfinalizer(lambda: fxt_clean_table(Project))
    repo = ProjectRepository(fxt_session)

    with pytest.raises(ResourceNotFoundError):
        repo.delete_project(uuid4())
