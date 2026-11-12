import test from 'node:test';
import assert from 'node:assert/strict';
import { configuredFieldOfView } from '../../seafront/web-static/src/field-of-view.js';

function config(objective, width = 3000, height = 3000) {
    return [{ handle: 'camera.main.objective', value: objective, options: [
        { handle: '4xolympus', info: { magnification: 4 } },
        { handle: '10xolympus', info: { magnification: 10 } },
        { handle: '20xolympus', info: { magnification: 20 } },
    ] },
        { handle: 'camera.main.image.width_px', value: width },
        { handle: 'camera.main.image.height_px', value: height }];
}
const near = (actual, expected) => assert(Math.abs(actual - expected) < 1e-12);

test('20x matches the calibrated 0.9 mm field at 3000 pixels', () => {
    const fov = configuredFieldOfView(config('20xolympus'));
    near(fov.fovx, 0.9);
    near(fov.fovy, 0.9);
});

test('objective magnification inversely scales the field, matching metadata pixel sizes', () => {
    near(configuredFieldOfView(config('10xolympus')).fovx, 1.8);
    near(configuredFieldOfView(config('4xolympus')).fovx, 4.5);
});

test('the configured rectangular crop determines width and height independently', () => {
    const fov = configuredFieldOfView(config('20xolympus', 2500, 2000));
    near(fov.fovx, 0.75);
    near(fov.fovy, 0.6);
});

test('unknown objectives and missing or invalid dimensions do not invent a field size', () => {
    for (const objective of [null, undefined, '', 'unknown', '0x', 'Olympus20x']) {
        assert.equal(configuredFieldOfView(config(objective)), null);
    }
    for (const width of [undefined, null, '', 0, -1, NaN, Infinity]) {
        assert.equal(configuredFieldOfView(config('20xolympus', width, 2000).map(item =>
            item.handle.endsWith('width_px') ? { ...item, value: width } : item)), null);
    }
    assert.equal(configuredFieldOfView([]), null);
    assert.equal(configuredFieldOfView(null), null);
});

test('objective handles are opaque: metadata determines the magnification', () => {
    const items = config('lens-a');
    items[0].options = [{ handle: 'lens-a', info: { magnification: 10 } }];
    near(configuredFieldOfView(items).fovx, 1.8);
    items[0].options = [{ handle: 'lens-a', info: null }];
    assert.equal(configuredFieldOfView(items), null);
    items[0].value = '20xolympus';
    items[0].options = [{ handle: '20xolympus', info: { magnification: 4 } }];
    near(configuredFieldOfView(items).fovx, 4.5);
});
