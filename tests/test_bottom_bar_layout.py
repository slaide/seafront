"""
The bottom bar keeps its height when a light turns on.

The bar's status column is min-content wide, so "Light on: Fluorescence 405 nm Ex 100%" wrapped
at every word, four lines tall, and each light switched on or off resized the bar and the page
above it. Measured in a browser against a mock seafront, with the light turned on the way the
page does it: by streaming a channel.

Needs playwright's chromium: `uv run playwright install chromium`.
"""

import pytest
from playwright.sync_api import sync_playwright

from tests.mock_server import mock_server

APP = "document.body._x_dataStack[0]"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with mock_server(tmp_path_factory) as srv:
        yield srv


@pytest.mark.parametrize("width", [1920, 1280])
def test_turning_a_light_on_keeps_the_bottom_bar_height(server, width):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(server.url)
            page.wait_for_function(f"{APP}?.initDone === true")
            bar = page.locator("#bottom-bar")
            lights = page.locator(".lights-indicator")
            height_off = bar.bounding_box()["height"]

            page.evaluate(f"""async () => {{
                const app = {APP};
                const channel = app.microscope_config.channels[0];
                channel.enabled = true;
                await app.focusToggleLive(channel);
            }}""")
            page.wait_for_function(
                "document.querySelector('.lights-indicator').textContent.startsWith('Light on')")
            height_on = bar.bounding_box()["height"]
            text = lights.text_content()

            page.evaluate(f"async () => {{ const app = {APP}; await app.buttons_endStreaming(); await app.lockRelease(); }}")
        finally:
            browser.close()

    assert height_on == pytest.approx(height_off), (
        f"the bar is {height_off:.0f} px tall with every light off, {height_on:.0f} px with '{text}'")
