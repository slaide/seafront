import test from 'node:test';
import assert from 'node:assert/strict';
import { wellGridHit, wellCornerRadius } from '../../seafront/web-static/src/well-selection.js';

const first = { left: 20, top: 30, width: 10, height: 20 };
const last = { left: 44, top: 74, width: 10, height: 20 };
const hit = (x, y, clamp = false) => wellGridHit(x, y, first, last, 3, 3, clamp);

test('gaps resolve to their nearest well, including horizontal/vertical gutter intersections', () => {
    assert.deepEqual(hit(30.5, 40), { col: 0, row: 0 });
    assert.deepEqual(hit(31.5, 40), { col: 1, row: 0 });
    assert.deepEqual(hit(37, 51.5), { col: 1, row: 1 });
    assert.deepEqual(hit(43.5, 73.5), { col: 2, row: 2 });
});

test('headers and space outside the well grid cannot begin a selection', () => {
    assert.equal(hit(19, 40), null);
    assert.equal(hit(25, 29), null);
    assert.equal(hit(55, 90), null);
    assert.equal(hit(40, 95), null);
});

test('captured drags clamp outside the plate so release still completes a range', () => {
    assert.deepEqual(hit(-200, 1000, true), { col: 0, row: 2 });
    assert.deepEqual(hit(1000, -200, true), { col: 2, row: 0 });
});

test('a single well and non-square cells map to valid indices', () => {
    assert.deepEqual(wellGridHit(25, 40, first, first, 1, 1), { col: 0, row: 0 });
    assert.deepEqual(wellGridHit(-1, 100, first, first, 1, 1, true), { col: 0, row: 0 });
});

const plate = (x, y, r) => ({ Well_size_x_mm: x, Well_size_y_mm: y, Well_edge_radius_mm: r });

test('wells whose corner radius is half their size draw as circles', () => {
    assert.equal(wellCornerRadius(plate(6.4, 6.4, 3.2)), '50% / 50%');      // revvity 96
    assert.equal(wellCornerRadius(plate(6.35, 6.35, 3.175)), '50% / 50%');  // corning 96
    assert.equal(wellCornerRadius(plate(34, 34, 17)), '50% / 50%');         // ibidi 6
});

test('small corner radii scale with the well, square wells keep square corners', () => {
    assert.equal(wellCornerRadius(plate(3.26, 3.26, 0.2)), '6.135% / 6.135%');
    assert.equal(wellCornerRadius(plate(75, 26, 0)), '0% / 0%');
});

test('non-square wells get round corners, not stretched ones', () => {
    // CSS takes horizontal percentages of the width and vertical ones of the height
    assert.equal(wellCornerRadius(plate(4, 2, 0.5)), '12.5% / 25%');
});

test('a radius beyond half the well is clamped to the half', () => {
    assert.equal(wellCornerRadius(plate(4, 2, 1.5)), '37.5% / 50%');
});
