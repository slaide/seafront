"""
The HTTP surface: a FastAPI app, the routers that make it up, and nothing else.

Data flow through this package is one-way:

    __main__  ->  server/app  ->  server/api/*  ->  server/core  ->  hardware/*

`app` knows about routers, routers know about Core, Core knows about the microscope. Nothing
points back up, which is why the schema can be derived from the route table and the route
table from the handlers.

/openapi.json is FastAPI's own, generated from the request and response models the routes are
annotated with. There is no second description of the API to keep in step: the schema a
client reads is derived from the models the server validates against.
"""

import faulthandler
import os
import pathlib as path
import signal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from seafront.logger import logger
from seafront.server import asyncapi
from seafront.server.api import acquisition, actions, info, lock, sockets
from seafront.server.api.routing import RefusedError
from seafront.server.api.tags import OPENAPI_TAGS, RouteTag
from seafront.server.core import Core

PACKAGE_DIR = path.Path(__file__).resolve().parent.parent
WEB_STATIC = PACKAGE_DIR / "web-static"


def make_app(core: Core | None = None) -> FastAPI:
    """
    build the server around a Core; the Core is reachable from any handler as CoreDep

    `core` is None only when something wants the app purely for its schema - scripts that
    generate the client types do that, so they describe the same app the server serves
    without needing a microscope to describe it.
    """

    # static file paths are written relative to the package, and so is anything else that
    # resolves a path at request time
    os.chdir(PACKAGE_DIR)

    app = FastAPI(
        debug=False,
        title="seafront api",
        version="0.2.0",
        description="Seafront OpenAPI schema",
        openapi_tags=OPENAPI_TAGS,
        # One schema per model, not a -Input/-Output pair. The pair only ever differs in
        # which fields carry a default, and splitting it would hand every client two names
        # for one thing (and the generated TypeScript two types for one model).
        separate_input_output_schemas=False,
    )
    if core is not None:
        app.state.core = core

    for module in (info, actions, acquisition, lock):
        app.include_router(module.api.router)
    app.include_router(info.router)
    app.include_router(sockets.router)

    @app.get("/", tags=[RouteTag.STATIC_FILES.value], operation_id="index.get")
    async def index() -> FileResponse:
        """the web UI"""
        return FileResponse(WEB_STATIC / "index.html")

    @app.get("/asyncapi.json", tags=[RouteTag.DOCUMENTATION.value], operation_id="asyncapi.json.get")
    async def asyncapi_json() -> dict:
        """
        the websocket contract, as AsyncAPI 3.0

        The sibling of /openapi.json: sockets have no place in an OpenAPI document, so the
        channels, their messages and which of them need the device lease are published here.
        """
        return asyncapi.build(
            title=f"{app.title} websockets",
            version=app.version,
            shared=app.openapi()["components"]["schemas"],
        )

    for mount in ("css", "src", "resources", "vendor"):
        app.mount(f"/{mount}", StaticFiles(directory=WEB_STATIC / mount), name=f"web-{mount}")

    @app.exception_handler(RefusedError)
    async def refused_handler(request: Request, exc: RefusedError):
        "400 and 421: flat {status, message}, the shape these refusals have always had"
        return JSONResponse(status_code=exc.status_code, content=exc.body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.debug(f"Validation error at {request.url}: {exc.errors()}")
        return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))

    # websocket compression is disabled in the uvicorn configuration: compression time is
    # unpredictable, 70ms for an all-white image and 1400ms for an all-black one, which is
    # not a rare case in practice
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # SIGUSR1 -> all thread backtraces on stderr. Core installs SIGUSR2 for the richer dump.
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    return app
