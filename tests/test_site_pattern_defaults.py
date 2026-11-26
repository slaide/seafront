"""
Sites whose center lies outside the well start excluded.

The well outline is a rectangle of Well_size_x_mm x Well_size_y_mm with corners rounded by
Well_edge_radius_mm, so a well whose radius is half its size is round. Site centers form a grid
centered on the well. The user's own choices survive later grid edits, except for a site the edit
moved across the outline, which takes the new default.

site-pattern.js is pure, so it is imported straight into node.
"""

import json
import math
import subprocess
from pathlib import Path

SITE_PATTERN = Path(__file__).resolve().parent.parent / "seafront" / "web-static" / "src" / "site-pattern.js"

PROBE = r"""
import { pathToFileURL } from "node:url";
const [modulePath, call] = process.argv.slice(2);
const module = await import(pathToFileURL(modulePath).href);
const { fn, args } = JSON.parse(call);
console.log(JSON.stringify(module[fn](...args)));
"""

IBIDI_6 = {"Well_size_x_mm": 34.0, "Well_size_y_mm": 34.0, "Well_edge_radius_mm": 17.0}


def _call(tmp_path: Path, fn: str, *args):
    probe = tmp_path / "probe.mjs"
    probe.write_text(PROBE)
    out = subprocess.run(
        ["node", str(probe), str(SITE_PATTERN), json.dumps({"fn": fn, "args": list(args)})],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(out.stdout)


def _grid(num_x, num_y, delta_x_mm, delta_y_mm, mask=()):
    return {"num_x": num_x, "num_y": num_y, "delta_x_mm": delta_x_mm, "delta_y_mm": delta_y_mm,
            "mask": list(mask)}


def _selected(mask):
    return {(s["col"], s["row"]) for s in mask if s["selected"]}


def _all_sites(num_x, num_y):
    return {(col, row) for col in range(num_x) for row in range(num_y)}


def test_a_round_well_excludes_sites_whose_center_lies_outside_the_circle(tmp_path):
    grid = _grid(5, 5, 8.0, 8.0)  # centers at -16, -8, 0, 8, 16 mm
    mask = _call(tmp_path, "siteMask", IBIDI_6, grid, None)
    inside = {(c, r) for c, r in _all_sites(5, 5) if math.hypot((c - 2) * 8, (r - 2) * 8) <= 17}
    assert len(mask) == 25
    assert _selected(mask) == inside
    assert len(inside) == 13  # the axes out to +-16 mm and the four (+-8, +-8) sites


def test_rounded_corners_exclude_corner_sites_that_a_sharp_corner_would_keep(tmp_path):
    grid = _grid(3, 3, 4.8, 4.8)  # corner centers at (+-4.8, +-4.8) in a 10 mm well
    sharp = {"Well_size_x_mm": 10.0, "Well_size_y_mm": 10.0, "Well_edge_radius_mm": 0.0}
    rounded = {**sharp, "Well_edge_radius_mm": 2.0}
    assert _selected(_call(tmp_path, "siteMask", sharp, grid, None)) == _all_sites(3, 3)
    assert _selected(_call(tmp_path, "siteMask", rounded, grid, None)) == {
        (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}


def test_a_rectangular_well_bounds_each_axis_by_its_own_size(tmp_path):
    holder = {"Well_size_x_mm": 75.0, "Well_size_y_mm": 26.0, "Well_edge_radius_mm": 0.0}
    grid = _grid(1, 5, 10.0, 10.0)  # y centers at -20, -10, 0, 10, 20 mm; the well reaches +-13
    assert _selected(_call(tmp_path, "siteMask", holder, grid, None)) == {(0, 1), (0, 2), (0, 3)}


def test_a_site_exactly_on_the_outline_counts_as_inside(tmp_path):
    well = {"Well_size_x_mm": 6.4, "Well_size_y_mm": 6.4, "Well_edge_radius_mm": 3.2}
    grid = _grid(3, 1, 3.2, 3.2)
    assert _selected(_call(tmp_path, "siteMask", well, grid, None)) == _all_sites(3, 1)


def test_an_unchanged_layout_keeps_every_choice_the_user_made(tmp_path):
    grid = _grid(5, 5, 8.0, 8.0)
    mask = _call(tmp_path, "siteMask", IBIDI_6, grid, None)
    for site in mask:
        if (site["col"], site["row"]) in {(2, 2), (0, 0)}:  # exclude the center, add a corner
            site["selected"] = not site["selected"]
    expected = _selected(mask)
    layout = {"plate": IBIDI_6, "grid": _grid(5, 5, 8.0, 8.0)}
    again = _call(tmp_path, "siteMask", IBIDI_6, {**grid, "mask": mask}, layout)
    assert _selected(again) == expected


def test_a_site_moved_across_the_outline_takes_the_new_default(tmp_path):
    small = _grid(3, 3, 8.0, 8.0)  # every center within 11.4 mm of the well center
    mask = _call(tmp_path, "siteMask", IBIDI_6, small, None)
    assert _selected(mask) == _all_sites(3, 3)
    for site in mask:
        if (site["col"], site["row"]) == (1, 1):
            site["selected"] = False

    wide = _grid(3, 3, 14.0, 14.0, mask)  # corners now 19.8 mm out, edges 14 mm
    mask = _call(tmp_path, "siteMask", IBIDI_6, wide, {"plate": IBIDI_6, "grid": small})
    assert _selected(mask) == {(1, 0), (0, 1), (2, 1), (1, 2)}  # center stays excluded

    narrow = _grid(3, 3, 8.0, 8.0, mask)
    mask = _call(tmp_path, "siteMask", IBIDI_6, narrow, {"plate": IBIDI_6, "grid": wide})
    assert _selected(mask) == _all_sites(3, 3) - {(1, 1)}


def test_new_sites_from_a_grid_resize_start_at_the_default(tmp_path):
    mask = _call(tmp_path, "siteMask", IBIDI_6, _grid(1, 1, 8.0, 8.0), None)
    grown = _grid(5, 5, 8.0, 8.0, mask)
    mask = _call(tmp_path, "siteMask", IBIDI_6, grown, {"plate": IBIDI_6, "grid": _grid(1, 1, 8.0, 8.0)})
    assert len(_selected(mask)) == 13
    assert (0, 0) not in _selected(mask)


def test_a_selection_of_unknown_layout_is_kept_as_is(tmp_path):
    # a loaded protocol: its selection was made for a layout this page never saw
    loaded = [{"col": c, "row": r, "selected": True} for c, r in sorted(_all_sites(5, 5))]
    grid = _grid(5, 5, 8.0, 8.0, loaded)
    assert _selected(_call(tmp_path, "siteMask", IBIDI_6, grid, None)) == _all_sites(5, 5)


def test_a_plate_without_well_dimensions_excludes_nothing(tmp_path):
    grid = _grid(3, 3, 8.0, 8.0)
    plate = {"Well_size_x_mm": None, "Well_size_y_mm": None, "Well_edge_radius_mm": None}
    assert _selected(_call(tmp_path, "siteMask", plate, grid, None)) == _all_sites(3, 3)
