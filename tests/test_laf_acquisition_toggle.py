"""
The protocol's "use laser autofocus during acquisition" setting stays reachable.

It used to be editable only in the focus reference setup, which shows below the reference
channel and nowhere else. A page reloaded with LAF on and no reference chosen had no way to turn
it off for an acquisition.

Checked on index.html as text: every checkbox bound to the setting is found, and its Alpine
`:disabled` expression is evaluated in node the way Alpine evaluates it, against the page state
of each case.
"""

import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "seafront" / "web-static" / "index.html"
SETTING = "microscope_config.autofocus_enabled"

EVALUATE = r"""
const [expression, scope] = JSON.parse(process.argv[1]);
const lookup = new Proxy(scope, { has: () => true, get: (target, key) => target[key] });
console.log(JSON.stringify(!!new Function("scope", `with (scope) { return (${expression}); }`)(lookup)));
"""


class _Toggles(HTMLParser):
    """checkboxes bound to SETTING, with the x-if conditions of the templates around them"""

    def __init__(self):
        super().__init__()
        self.conditions: list[str | None] = []
        self.toggles: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "template":
            self.conditions.append(attributes.get("x-if"))
        bound = [value for name, value in attrs if name.startswith("x-model")]
        if tag == "input" and attributes.get("type") == "checkbox" and SETTING in bound:
            self.toggles.append({
                "inside_x_if": [c for c in self.conditions if c],
                "disabled": "disabled" in attributes,
                "disabled_when": attributes.get(":disabled") or attributes.get("x-bind:disabled"),
            })

    def handle_endtag(self, tag):
        if tag == "template" and self.conditions:
            self.conditions.pop()


def _toggles() -> list[dict]:
    parser = _Toggles()
    parser.feed(INDEX_HTML.read_text())
    return parser.toggles


def _disabled(toggle: dict, scope: dict) -> bool:
    if toggle["disabled"]:
        return True
    if toggle["disabled_when"] is None:
        return False
    out = subprocess.run(
        ["node", "-e", EVALUATE, json.dumps([toggle["disabled_when"], scope])],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(out.stdout)


def _page(*, calibrated: bool, enabled: bool) -> dict:
    return {
        "laserAutofocusIsCalibrated": calibrated,
        "microscope_config": {"autofocus_enabled": enabled},
        "focusReferenceChannelHandle": None,
        "state": {"is_busy": False},
    }


def _always_shown() -> list[dict]:
    "toggles that render without a focus reference (the whole page sits in x-if='initDone')"
    shown = [t for t in _toggles() if not any("focusReference" in c for c in t["inside_x_if"])]
    assert shown, f"every checkbox for {SETTING} needs a focus reference to show: {_toggles()}"
    return shown


def test_laf_can_be_turned_off_without_a_focus_reference():
    for toggle in _always_shown():
        assert not _disabled(toggle, _page(calibrated=True, enabled=True)), toggle


def test_laf_can_be_turned_off_even_when_it_is_not_calibrated():
    # a protocol can carry the setting onto a machine whose calibration was reset
    for toggle in _always_shown():
        assert not _disabled(toggle, _page(calibrated=False, enabled=True)), toggle


def test_laf_can_be_turned_on_once_calibrated_but_not_before():
    for toggle in _always_shown():
        assert not _disabled(toggle, _page(calibrated=True, enabled=False)), toggle
        assert _disabled(toggle, _page(calibrated=False, enabled=False)), toggle


def test_every_toggle_for_the_setting_follows_the_same_rule():
    # the one in the focus reference setup too, so the two can never disagree
    for toggle in _toggles():
        assert not _disabled(toggle, _page(calibrated=False, enabled=True)), toggle
        assert _disabled(toggle, _page(calibrated=False, enabled=False)), toggle
