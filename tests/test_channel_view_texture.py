"""
The channel view draws exactly the pixels the image socket sent.

/ws/get_info/acquired_image reports the dimensions of the transmitted (already downsampled)
image. channelview.js used to halve them again, so a 1250x1250 frame became a 625x625 texture:
every real row then filled two texture rows (left half, right half), which showed as horizontal
stripes over the top half of the image.

channelview.js is loaded in node with `three` mapped to the vendored copy, the same mapping the
page's import map makes.
"""

import json
import subprocess
from pathlib import Path

WEB_STATIC = Path(__file__).resolve().parent.parent / "seafront" / "web-static"

PROBE = r"""
import { register } from "node:module";
import { pathToFileURL } from "node:url";
const [staticDir, w, h, bits] = process.argv.slice(2);
const three = pathToFileURL(`${staticDir}/vendor/three/three-0-176-0.js`).href;
register("data:text/javascript," + encodeURIComponent(
    `export async function resolve(s, c, n) {
        return s === "three" ? { url: ${JSON.stringify(three)}, shortCircuit: true } : n(s, c);
    }`));
const { ChannelImageView } = await import(pathToFileURL(`${staticDir}/src/channelview.js`).href);
const width = Number(w), height = Number(h), bit_depth = Number(bits);
const view = Object.create(ChannelImageView.prototype);
const img = view._makeImage({
    width, height, bit_depth, data: new ArrayBuffer(width * height * (bit_depth / 8)),
});
console.log(JSON.stringify({
    texture: [img.texture.image.width, img.texture.image.height],
    image: [img.width, img.height],
    pixels: img.data.length,
}));
"""


def _make_image(tmp_path: Path, width: int, height: int, bit_depth: int) -> dict:
    probe = tmp_path / "probe.mjs"
    probe.write_text(PROBE)
    out = subprocess.run(
        ["node", str(probe), str(WEB_STATIC), str(width), str(height), str(bit_depth)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(out.stdout)


def test_the_texture_has_the_transmitted_image_dimensions(tmp_path):
    # 2500x2500 camera frame, sent at the UI's downsample factor 2
    made = _make_image(tmp_path, 1250, 1250, 8)
    assert made["texture"] == [1250, 1250]
    assert made["image"] == [1250, 1250]
    assert made["pixels"] == 1250 * 1250


def test_a_16_bit_texture_has_the_transmitted_image_dimensions(tmp_path):
    made = _make_image(tmp_path, 2048, 1536, 16)
    assert made["texture"] == [2048, 1536]
    assert made["pixels"] == 2048 * 1536
