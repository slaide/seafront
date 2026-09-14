"""
/asyncapi.json: the websocket half of the API, generated the same way the openapi half is.

Sockets cannot be described by OpenAPI - it has no vocabulary for "the server pushes one of
these until you go away" - so they are published as an AsyncAPI 3.0 document instead. Nothing
here is written by hand: channels come from the `SocketSpec` each handler declares, and the
message schemas are the Pydantic models those handlers actually parse and send, so the
document cannot describe a message the code does not use.

`x-requires-device-lease` means the same thing it does in /openapi.json: this channel drives
the hardware, so the handshake must carry a `lock_token`.
"""

import typing as tp

from seafront.server.api.socketspec import SOCKETS, Frame, SocketSpec

ASYNCAPI_VERSION = "3.0.0"

type Schemas = dict[str, tp.Any]


def _message(frame: Frame, schemas: Schemas, shared: Schemas) -> dict[str, tp.Any]:
    """one AsyncAPI message, with its payload schema hoisted into components/schemas"""
    message: dict[str, tp.Any] = {"name": frame.name, "summary": frame.description}

    if frame.model is None:
        message["payload"] = frame.schema if frame.schema else {}
        return message

    # $defs from a model's schema become shared components, exactly as openapi does it
    schema = frame.model.model_json_schema(
        ref_template="#/components/schemas/{model}", mode="serialization"
    )
    defs = schema.pop("$defs", {})
    for name, definition in {**defs, frame.model.__name__: schema}.items():
        # A model the HTTP surface already publishes keeps that definition. The two documents
        # describe the same Pydantic classes, and pydantic renders a model slightly
        # differently depending on the direction it is asked about; taking openapi's copy is
        # what keeps one model from having two descriptions.
        schemas[name] = shared.get(name, definition)
    message["payload"] = {"$ref": f"#/components/schemas/{frame.model.__name__}"}
    message["title"] = frame.model.__name__
    return message


def _channel(spec: SocketSpec, schemas: Schemas, shared: Schemas) -> dict[str, tp.Any]:
    messages = {
        frame.name: _message(frame, schemas, shared) for frame in (*spec.receive, *spec.send)
    }
    channel: dict[str, tp.Any] = {
        "address": spec.path,
        "title": spec.summary,
        "description": spec.description,
        "messages": messages,
        "x-requires-device-lease": spec.requires_lease,
    }
    if spec.close_codes:
        channel["x-close-codes"] = {str(code): why for code, why in sorted(spec.close_codes.items())}
    return channel


def build(title: str, version: str, shared: Schemas | None = None) -> dict[str, tp.Any]:
    """
    the whole document, derived from what the socket handlers declared at import time

    `shared` is the openapi document's components/schemas. Passing it makes the two documents
    reference one description per model instead of two that can disagree.
    """
    shared = shared or {}
    schemas: Schemas = {}
    channels: dict[str, tp.Any] = {}
    operations: dict[str, tp.Any] = {}

    for spec in SOCKETS:
        key = spec.path.removeprefix("/ws/").replace("/", ".")
        channels[key] = _channel(spec, schemas, shared)

        # AsyncAPI names operations from the application's point of view: it *receives* what
        # the client sends, and *sends* what the client reads.
        for action, frames in (("receive", spec.receive), ("send", spec.send)):
            if not frames:
                continue
            operations[f"{key}.{action}"] = {
                "action": action,
                "channel": {"$ref": f"#/channels/{key}"},
                "messages": [
                    {"$ref": f"#/channels/{key}/messages/{frame.name}"} for frame in frames
                ],
            }

    return {
        "asyncapi": ASYNCAPI_VERSION,
        "info": {
            "title": title,
            "version": version,
            "description": "websocket channels of the seafront microscope server",
        },
        "defaultContentType": "application/json",
        "channels": channels,
        "operations": operations,
        "components": {"schemas": schemas},
    }


__all__ = ["ASYNCAPI_VERSION", "build"]
