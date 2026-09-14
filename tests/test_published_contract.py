"""
The two published documents have to be usable, not just present.

/openapi.json and /asyncapi.json are what a client generates code from. Both are derived -
from the route table and from the socket declarations - which is what stops them describing
an API the server does not have. It does not stop them being internally broken: a `$ref` to
a schema nobody hoisted, an operation pointing at a channel that is not there, a message
whose payload never made it into components. A generator hits those; nothing else does.

Everything here reads the documents the server actually serves, built in-process so it costs
nothing. `tests/test_route_policy.py` checks what the documents *say* about policy against
what the server *does*; this checks that what they say is well-formed.
"""

import re

import pytest

from seafront.server import asyncapi
from seafront.server.app import make_app

REF = re.compile(r'"\$ref":\s*"([^"]+)"')


@pytest.fixture(scope="module")
def openapi() -> dict:
    """the HTTP document, from an app with no microscope behind it"""
    return make_app(None).openapi()


@pytest.fixture(scope="module")
def async_doc(openapi) -> dict:
    return asyncapi.build(
        title="test", version="0", shared=openapi["components"]["schemas"]
    )


def refs(document: dict) -> set[str]:
    """every $ref anywhere in the document"""
    import json

    return set(REF.findall(json.dumps(document)))


def resolve(document: dict, ref: str) -> object:
    """walk a local $ref, or raise KeyError naming the step that was missing"""
    assert ref.startswith("#/"), f"not a local ref: {ref}"
    node: object = document
    for step in ref.removeprefix("#/").split("/"):
        step = step.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or step not in node:
            raise KeyError(f"{ref} breaks at {step!r}")
        node = node[step]
    return node


class TestOpenapi:
    def test_every_ref_resolves(self, openapi):
        broken = []
        for ref in refs(openapi):
            try:
                resolve(openapi, ref)
            except KeyError as e:
                broken.append(str(e))
        assert not broken, broken

    def test_every_route_has_an_operation_id(self, openapi):
        """clients name their generated methods from it, so a missing one is a broken client"""
        for path, ops in openapi["paths"].items():
            for method, op in ops.items():
                assert op.get("operationId"), f"{method.upper()} {path}"

    def test_operation_ids_are_unique(self, openapi):
        seen: dict[str, str] = {}
        for path, ops in openapi["paths"].items():
            for method, op in ops.items():
                key = op["operationId"]
                assert key not in seen, f"{key} used by both {seen[key]} and {method.upper()} {path}"
                seen[key] = f"{method.upper()} {path}"

    def test_every_post_route_documents_its_refusals(self, openapi):
        """
        409 and 500 can come back from any route that reaches the handler

        A client that does not know a status can come back does not handle it, and these two
        are the ones every command shares.
        """
        for path, ops in openapi["paths"].items():
            post = ops.get("post")
            if post is None:
                continue
            for status in ("409", "500"):
                assert status in post["responses"], f"{path} does not document {status}"

    def test_no_model_is_split_into_an_input_and_output_pair(self, openapi):
        """
        separate_input_output_schemas=False, held to it

        The pair only ever differs in which fields carry a default, and splitting it hands
        every client two names for one thing.
        """
        split = [
            name
            for name in openapi["components"]["schemas"]
            if name.endswith("-Input") or name.endswith("-Output")
        ]
        assert not split, split


class TestAsyncapi:
    def test_every_ref_resolves(self, async_doc):
        broken = []
        for ref in refs(async_doc):
            try:
                resolve(async_doc, ref)
            except KeyError as e:
                broken.append(str(e))
        assert not broken, broken

    def test_every_channel_has_messages(self, async_doc):
        for key, channel in async_doc["channels"].items():
            assert channel["messages"], f"channel {key} declares no messages"

    def test_every_channel_opens_with_a_handshake(self, async_doc):
        """
        the first message on every socket is the handshake the gate reads

        A channel that does not declare it is a channel whose client will not send one, and
        the gate closes 4400 on the message it does send instead.
        """
        for key, channel in async_doc["channels"].items():
            assert "handshake" in channel["messages"], f"channel {key} declares no handshake"

    def test_every_operation_points_at_a_channel_that_exists(self, async_doc):
        for name, operation in async_doc["operations"].items():
            resolve(async_doc, operation["channel"]["$ref"])
            for message in operation["messages"]:
                resolve(async_doc, message["$ref"])
            assert operation["action"] in {"send", "receive"}, name

    def test_a_gated_channel_documents_the_close_code_it_uses(self, async_doc):
        for key, channel in async_doc["channels"].items():
            if channel.get("x-requires-device-lease"):
                assert "4423" in channel.get("x-close-codes", {}), key

    def test_channels_are_addressed_under_ws(self, async_doc):
        for key, channel in async_doc["channels"].items():
            assert channel["address"].startswith("/ws/"), f"{key}: {channel['address']}"


class TestTheTwoDocumentsAgree:
    def test_a_shared_model_has_one_description(self, openapi, async_doc):
        """
        both documents describe the same Pydantic classes

        Pydantic renders a model slightly differently depending on which direction it is
        asked about, so a model published by both could otherwise arrive at a client as two
        incompatible types with the same name.
        """
        http = openapi["components"]["schemas"]
        sockets = async_doc["components"]["schemas"]
        disagreements = [name for name in set(http) & set(sockets) if http[name] != sockets[name]]
        assert not disagreements, disagreements

    def test_the_lease_flag_means_the_same_thing_in_both(self, openapi, async_doc):
        """one name for one idea, whichever document a client reads"""
        http_flags = {
            op.get("x-requires-device-lease")
            for ops in openapi["paths"].values()
            for op in ops.values()
            if "x-requires-device-lease" in op
        }
        socket_flags = {c["x-requires-device-lease"] for c in async_doc["channels"].values()}
        assert http_flags == {True, False}, "the http surface should have both gated and open routes"
        assert socket_flags <= {True, False}
