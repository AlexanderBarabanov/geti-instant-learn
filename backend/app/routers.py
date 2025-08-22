# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

from fastapi import APIRouter

pipelines_router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
state_router = APIRouter(prefix="/state", tags=["State"])
