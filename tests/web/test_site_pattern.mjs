import test from 'node:test';
import assert from 'node:assert/strict';
import { sitePatternGeometry, sitePatternSvg } from '../../seafront/web-static/src/site-pattern.js';

const plate = { Well_size_x_mm: 4, Well_size_y_mm: 2, Well_edge_radius_mm: 0.2 };
const fov = { fovx: 0.8, fovy: 0.4 };
function grid(nx, ny, dx = 1, dy = 0.5) {
    return { num_x: nx, num_y: ny, delta_x_mm: dx, delta_y_mm: dy,
        mask: Array.from({ length: nx * ny }, (_, i) => ({ col: i % nx, row: Math.floor(i / nx), selected: true })) };
}

test('a single camera footprint sits at the well center with correct physical aspect ratios', () => {
    const { well, sites } = sitePatternGeometry(plate, grid(1, 1), fov);
    assert.equal(well.width / well.height, 2);
    assert.equal(sites[0].cx, 160);
    assert.equal(sites[0].cy, 160);
    assert.equal(sites[0].width / well.width, 0.2);
    assert.equal(sites[0].height / well.height, 0.2);
});

test('even and odd site grids remain centered with row zero above later rows', () => {
    const { sites, well } = sitePatternGeometry(plate, grid(2, 3), fov);
    assert.equal((sites[0].cx + sites[1].cx) / 2, 160);
    assert.equal(sites[2].cy, 160);
    assert(sites[0].cy < sites[2].cy);
    assert(sites[2].cy < sites[4].cy);
    assert.equal(sites[1].cx - sites[0].cx, well.width / 4);
});

test('footprints extending beyond a well remain fully visible', () => {
    const { well, sites } = sitePatternGeometry(plate, grid(5, 5, 2, 2), fov);
    assert(sites[0].x < well.x);
    for (const site of sites) {
        assert(site.x >= 26 && site.y >= 26);
        assert(site.x + site.width <= 294 && site.y + site.height <= 294);
    }
});

test('excluded and overlapping sites retain individually accessible center targets', () => {
    const pattern = grid(3, 2, 0.1, 0.1);
    pattern.mask[2].selected = false;
    const markup = sitePatternSvg(sitePatternGeometry(plate, pattern, fov));
    assert.equal((markup.match(/role="button"/g) || []).length, 6);
    assert(markup.includes('aria-label="Site 3, 1" aria-pressed="false"'));
    assert(markup.lastIndexOf('class="site-footprint') < markup.indexOf('class="site-center '));
});

test('incomplete numeric input does not produce invalid SVG coordinates', () => {
    assert.equal(sitePatternGeometry(plate, { ...grid(1, 1), num_x: NaN }, fov), null);
    assert.equal(sitePatternSvg(null), '');
});
