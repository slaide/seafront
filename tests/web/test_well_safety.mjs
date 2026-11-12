import test from 'node:test';
import assert from 'node:assert/strict';
import { wellForbiddenArea } from '../../seafront/web-static/src/well-safety.js';

// Deliberately asymmetric geometry catches row inversion and omitted half-well offsets.
const plate = { Offset_A1_x_mm: 10, Offset_A1_y_mm: 20, Well_distance_x_mm: 5,
    Well_distance_y_mm: 6, Well_size_x_mm: 4, Well_size_y_mm: 2, Num_wells_y: 16 };
const area = { name: 'Clamp', min_x_mm: 17, max_x_mm: 22, min_y_mm: 27, max_y_mm: 33 };

test('well centers use physical rows and include the half-well offset', () => {
    assert.equal(wellForbiddenArea(plate, { row: 1, col: 1 }, [area]), area);
    assert.equal(wellForbiddenArea(plate, { row: 2, col: 2 }, [area]), area);
    assert.equal(wellForbiddenArea(plate, { row: 0, col: 1 }, [area]), undefined);
});

test('area boundaries are inclusive, matching server validation', () => {
    const point = { name: 'Point', min_x_mm: 12, max_x_mm: 12, min_y_mm: 21, max_y_mm: 21 };
    assert.equal(wellForbiddenArea(plate, { row: 0, col: 0 }, [point]), point);
    assert.equal(wellForbiddenArea(plate, { row: 0, col: 1 }, [point]), undefined);
});

test('a well edge intersecting an area does not forbid an allowed center', () => {
    const edge = { name: 'Edge', min_x_mm: 10, max_x_mm: 11, min_y_mm: 20, max_y_mm: 22 };
    assert.equal(wellForbiddenArea(plate, { row: 0, col: 0 }, [edge]), undefined);
});

test('headers and plates without forbidden regions remain unrestricted', () => {
    assert.equal(wellForbiddenArea(plate, { row: -1, col: 1 }, [area]), undefined);
    assert.equal(wellForbiddenArea(plate, { row: 1, col: -1 }, [area]), undefined);
    assert.equal(wellForbiddenArea(plate, { row: 1, col: 1 }, []), undefined);
});
