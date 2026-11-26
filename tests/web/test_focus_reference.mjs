import test from 'node:test';
import assert from 'node:assert/strict';
import { commitFocusReference, restoreFocusReference } from '../../seafront/web-static/src/focus-reference.js';

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

/** @param {Record<string, string|number>} values */
const machineConfig = values => Object.entries(values).map(([handle, value]) => ({
    handle, name: '', value_kind: typeof value === 'number' ? 'float' : 'text', value,
}));

const calibrated = {
    'laser.autofocus.calibration.is_calibrated': 'yes',
    'laser.autofocus.calibration.ref_channel': 'fluo488',
    'laser.autofocus.calibration.ref_z_mm': 3.1,
    'laser.autofocus.calibration.ref_x_mm': 12.5,
    'laser.autofocus.calibration.ref_y_mm': 8.25,
};
const channels = [{ handle: 'bfled' }, { handle: 'fluo488' }];

test('a reloaded page gets the reference back from the calibration', () => {
    // The protocol was set up and LAF calibrated with fluo488 in focus, then the page reloaded.
    const state = restoreFocusReference(/** @type {any} */ (machineConfig(calibrated)), /** @type {any} */ (channels));
    assert.ok(state);
    assert.equal(state.focusReferenceChannelHandle, 'fluo488');
    assert.equal(state.focusReferenceZ, 3.1);
    assert.deepEqual(state.focusReferenceXY, { x: 12.5, y: 8.25 });
    assert.deepEqual([...state.focusConfirmedChannels], ['fluo488']);
});

test('nothing is restored without a calibration', () => {
    const config = machineConfig({ ...calibrated, 'laser.autofocus.calibration.is_calibrated': 'no' });
    assert.equal(restoreFocusReference(/** @type {any} */ (config), /** @type {any} */ (channels)), null);
});

test('a calibration that does not name its channel restores nothing', () => {
    // made before the channel was recorded: any channel picked here would be a guess
    const config = machineConfig({ ...calibrated, 'laser.autofocus.calibration.ref_channel': '' });
    assert.equal(restoreFocusReference(/** @type {any} */ (config), /** @type {any} */ (channels)), null);
});

test('a reference channel this microscope does not have is not restored', () => {
    const config = machineConfig({ ...calibrated, 'laser.autofocus.calibration.ref_channel': 'fluo730' });
    assert.equal(restoreFocusReference(/** @type {any} */ (config), /** @type {any} */ (channels)), null);
});
