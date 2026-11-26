"""
A machine config change made in the page survives a page reload.

The page keeps the protocol, machine config included, in localStorage. On reload it rebuilt the
machine config from the server's list alone, so a value the server had not been sent yet (it
is sent before the next snap, stream or autofocus action) or had forgotten (the value lives in
server memory and a seafront restart resets it) went back to the server's: e.g.
laser.autofocus.use_right_dot, which picks the dot the laser autofocus calibration tracks.

Privileged values (calibration, identity) stay the server's; tests/web covers that rule in
mergeMachineConfig. Needs playwright's chromium: `uv run playwright install chromium`.
"""

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from tests.mock_server import mock_server

APP = "document.body._x_dataStack[0]"
HANDLE = "laser.autofocus.use_right_dot"
SAVED = f"""() => {APP}.getMicroscopeCache().microscope_config?.microscope_config
    ?.machine_config?.find(i => i.handle === {HANDLE!r})?.value"""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with mock_server(tmp_path_factory) as srv:
        yield srv


def _value(page, handle: str) -> str:
    return page.evaluate(f"{APP}.getMachineConfigItem({handle!r})?.value")


def _saved_after_change(page, expected: str) -> str | None:
    "the page saves a change 50 ms after it; give it a few seconds"
    try:
        page.wait_for_function(f"({SAVED})() === {expected!r}", timeout=5000)
    except PlaywrightTimeoutError:
        pass
    return page.evaluate(SAVED)


def test_a_machine_config_change_survives_a_reload(server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(server.url)
            page.wait_for_function(f"{APP}?.initDone === true")
            assert _value(page, HANDLE) == "yes"

            page.evaluate(f"{APP}.getMachineConfigItem({HANDLE!r}).value = 'no'")
            assert _saved_after_change(page, "no") == "no"

            page.reload()
            page.wait_for_function(f"{APP}?.initDone === true")
            after_reload = _value(page, HANDLE)
        finally:
            browser.close()

    assert after_reload == "no"


def test_a_change_made_as_soon_as_the_page_shows_is_saved(server):
    """
    The page shows once the protocol is loaded and goes on loading, e.g. the list of stored
    protocols. Changes made meanwhile were not saved, so a reload lost them. Here the protocol
    list is held back until the change is checked.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            held = []
            page.route("**/api/acquisition/config_list", lambda route: held.append(route))
            page.goto(server.url)
            page.wait_for_function(f"{APP}?.initDone === true")

            page.evaluate(f"{APP}.getMachineConfigItem({HANDLE!r}).value = 'no'")
            saved = _saved_after_change(page, "no")
            assert held, "the protocol list was not requested; this test no longer holds the page in its loading phase"
        finally:
            browser.close()

    assert saved == "no"
