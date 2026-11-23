import test from 'node:test';
import assert from 'node:assert/strict';
import { commitFocusReference } from '../../seafront/web-static/src/focus-reference.js';

const position = { x_pos_mm: 12.5, y_pos_mm: 8.25, z_pos_mm: 3.1 };

test('setting the reference zeroes an offset left on the reference channel', () => {
    // The channel had an offset from when another channel was the reference. Snap all channels
    // with autofocus moved it 4.2 µm off the focus the LAF was just calibrated on.
    const reference = { handle: 'fluo488', enabled: true, z_offset_um: 4.2 };
    commitFocusReference(/** @type {any} */ (reference), position);
    assert.equal(reference.z_offset_um, 0);
});

test('the reference is recorded at the calibration position and counts as confirmed', () => {
    const reference = { handle: 'bfled', enabled: true, z_offset_um: -1.5 };
    const state = commitFocusReference(/** @type {any} */ (reference), position);
    assert.equal(state.focusReferenceZ, 3.1);
    assert.deepEqual(state.focusReferenceXY, { x: 12.5, y: 8.25 });
    assert.deepEqual([...state.focusConfirmedChannels], ['bfled']);
    assert.equal(state.focusMeasuredOffsetUM, null);
    assert.equal(state.laserAutofocusMeasuredOffset, null);
});
