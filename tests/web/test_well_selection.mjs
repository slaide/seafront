import test from 'node:test';
import assert from 'node:assert/strict';
import { wellGridHit } from '../../seafront/web-static/src/well-selection.js';

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
