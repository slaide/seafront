"""
The transport envelope: the fields a request carries that are about routing and access
rather than about the microscope.

A request model is `Envelope` plus the domain model, so the wire stays flat while the two
halves stay separate types. `command_of` is the boundary in the other direction: it takes
the envelope back off, so the hardware layer only ever sees a command that knows nothing
about HTTP.
"""

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    """fields every /api request carries, regardless of what it asks the microscope to do"""

    target_device: str = Field(..., description="Target device name for request routing")
    lock_token: str | None = Field(
        default=None,
        description="Device lease token from /api/lock/acquire; required while another client holds the lease",
    )


ENVELOPE_FIELDS: frozenset[str] = frozenset(Envelope.model_fields)


def command_of[C: BaseModel](body: Envelope, command: type[C]) -> C:
    """
    The domain command carried by a request, with the envelope stripped.

    `body` is a subclass of `command` (that is how the flat wire format is built), so every
    field the command needs has already been validated by the time this runs.
    """
    return command.model_validate({k: v for k, v in body if k not in ENVELOPE_FIELDS})
