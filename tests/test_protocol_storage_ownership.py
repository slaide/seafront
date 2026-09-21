import asyncio
import typing as tp

from seafront.server.protocol import ProtocolGenerator


class RecordingStore:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_closing_protocol_generator_closes_its_writer(monkeypatch):
    """Cancellation cleanup belongs to the protocol, not to its command executor."""
    inner_closed = False

    async def commands(
        _self: ProtocolGenerator,
    ) -> tp.AsyncGenerator[str]:
        nonlocal inner_closed
        try:
            yield "ready"
            yield "never reached"
        finally:
            inner_closed = True

    monkeypatch.setattr(ProtocolGenerator, "_generate_commands", commands)
    store = RecordingStore()
    protocol = object.__new__(ProtocolGenerator)
    object.__setattr__(protocol, "__dict__", {"image_store": store})
    object.__setattr__(protocol, "__pydantic_fields_set__", {"image_store"})
    object.__setattr__(protocol, "__pydantic_extra__", None)
    object.__setattr__(protocol, "__pydantic_private__", {})

    async def cancel() -> None:
        generator = protocol.generate()
        assert await anext(generator) == "ready"
        await generator.aclose()

    asyncio.run(cancel())

    assert inner_closed
    assert store.close_calls == 1
