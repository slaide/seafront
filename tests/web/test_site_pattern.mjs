import test from 'node:test';
import assert from 'node:assert/strict';
import { sitePatternGeometry, sitePatternSvg, sitePatternHit, siteRangeContains } from '../../seafront/web-static/src/site-pattern.js';

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

test('drag hit testing resolves gaps to the nearest center and clamps outside the view', () => {
    const geometry = sitePatternGeometry(plate, grid(3, 3, 1.2, 0.8), fov);
    const first = geometry.sites[0], second = geometry.sites[1];
    const x = (first.x + first.width + second.x) / 2;
    assert.equal(sitePatternHit(geometry, x - 0.1, first.cy).col, 0);
    assert.equal(sitePatternHit(geometry, x + 0.1, first.cy).col, 1);
    assert.equal(sitePatternHit(geometry, -100, -100), null);
    assert.equal(sitePatternHit(geometry, -100, -100, true).index, 0);
    assert.equal(sitePatternHit(geometry, 1000, 1000, true).index, 8);
    assert.equal(sitePatternHit(null, 0, 0), null);
});

test('reversed rectangular ranges include their endpoints without including other sites', () => {
    const range = { from: { col: 2, row: 2 }, to: { col: 0, row: 1 } };
    assert(siteRangeContains({ col: 1, row: 1 }, range));
    assert(siteRangeContains({ col: 2, row: 2 }, range));
    assert.equal(siteRangeContains({ col: 1, row: 0 }, range), false);
    const geometry = sitePatternGeometry(plate, grid(3, 3), fov);
    const markup = sitePatternSvg(geometry, { ...range, selected: false });
    assert.equal((markup.match(/class="site-footprint [^"]*range-remove/g) || []).length, 6);
    assert.equal((markup.match(/class="site-removal-mark"/g) || []).length, 6);
    assert(geometry.sites.every(site => site.selected), 'preview does not mutate the acquisition mask');
});
