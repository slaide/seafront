"""the /docs structure: which group of the api a route belongs to"""

from enum import StrEnum


class RouteTag(StrEnum):
    STATIC_FILES = "Static Files"
    ACTIONS = "Microscope Actions"
    ACQUISITION_CONTROLS = "Acquisition Controls"
    LOCK = "Device Lock"
    DOCUMENTATION = "Documentation"
    INFO = "Server Info"


OPENAPI_TAGS = [
    {"name": RouteTag.STATIC_FILES.value, "description": "serve static files of all sorts"},
    {
        "name": RouteTag.ACTIONS.value,
        "description": "actions for the microscope to perform immediately",
    },
    {
        "name": RouteTag.ACQUISITION_CONTROLS.value,
        "description": "acquisition related controls and information i/o",
    },
    {"name": RouteTag.LOCK.value, "description": "the device lease: one client at a time"},
    {"name": RouteTag.DOCUMENTATION.value, "description": "documentation on software and api"},
    {"name": RouteTag.INFO.value, "description": "server health and status information"},
]
