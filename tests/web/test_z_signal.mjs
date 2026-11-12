import test from 'node:test';
import assert from 'node:assert/strict';
import { imageSignalStats, emptySignalSeries, appendSignalFrame, signalGeometry, jogTarget } from '../../seafront/web-static/src/z-signal.js';

function frame(id, overrides = {}) {
    const channel = { handle: 'fluo488', exposure_time_ms: 10, analog_gain: 0, illum_perc: 20, filter_handle: null };
    return {
        data: new Uint16Array([10, 20, 30, 40]).buffer, width: 2, height: 2, bit_depth: 16, camera_bit_depth: 16,
        info: { image_id: String(id), timestamp: id, channel, position: { position: { x_pos_mm: 1, y_pos_mm: 2, z_pos_mm: 9 } } },
        capture: { image_id: String(id), timestamp: id, channel, position: { x_pos_mm: 1, y_pos_mm: 2, z_pos_mm: 1 }, pixel_format: 'mono16' },
        ...overrides,
    };
}

test('percentiles reject an isolated hot pixel while max and saturation retain it', () => {
    const pixels = new Uint16Array(10000).fill(100); pixels[9999] = 65535;
    const stats = imageSignalStats(pixels.buffer, 16, 16);
    assert.deepEqual(stats, { p99: 100, p999: 100, p9999: 100, max: 65535, saturated: 0.0001, pixels: 10000 });
    pixels.fill(300, 9980);
    assert.equal(imageSignalStats(pixels.buffer, 16, 16).p999, 300);
});

test('8-bit and native/padded 12-bit saturation count the actual camera ceiling', () => {
    assert.equal(imageSignalStats(new Uint8Array([0, 255]).buffer, 8, 8).saturated, 0.5);
    assert.equal(imageSignalStats(new Uint16Array([0, 4095, 2048]).buffer, 16, 12).saturated, 1 / 3);
    assert.equal(imageSignalStats(new Uint16Array([0, 65520, 32768]).buffer, 16, 12).saturated, 1 / 3);
    assert.equal(imageSignalStats(new Uint16Array([0, 4048]).buffer, 16, 12).saturated, 0);
    assert.equal(imageSignalStats(new Uint16Array([0, 65535]).buffer, 16, 12).saturated, 0.5);
    assert.throws(() => imageSignalStats(new ArrayBuffer(0), 8, 8), /Empty/);
    assert.throws(() => imageSignalStats(new ArrayBuffer(2), 12, 12), /Unsupported/);
});

test('trace uses capture Z/settings rather than a later status or mutable UI channel', () => {
    const series = appendSignalFrame(emptySignalSeries(), frame(1));
    assert.equal(series.samples[0].z, 1000);
    const next = frame(2); next.info.channel = { ...next.info.channel, exposure_time_ms: 99 };
    assert.equal(appendSignalFrame(series, next).segment, 0);
    assert.equal(appendSignalFrame(series, frame(1)), series);
    assert.equal(appendSignalFrame(series, frame(0)), series);
});

test('captured settings, XY, preview dimensions and bit-depth changes create separate segments', () => {
    const base = appendSignalFrame(emptySignalSeries(), frame(1));
    for (const field of ['exposure_time_ms', 'analog_gain', 'illum_perc', 'filter_handle']) {
        const next = frame(2); next.capture.channel = { ...next.capture.channel, [field]: field === 'filter_handle' ? 'new' : 25 };
        assert.equal(appendSignalFrame(base, next).segment, 1, field);
    }
    const moved = frame(2); moved.capture.position.x_pos_mm += 0.01;
    assert.equal(appendSignalFrame(base, moved).segment, 1);
    assert.equal(appendSignalFrame(base, frame(2, { width: 1 })).segment, 1);
    assert.equal(appendSignalFrame(base, frame(2, { camera_bit_depth: 12 })).segment, 1);
    const jogged = frame(2); jogged.capture.position.z_pos_mm += 0.001;
    assert.equal(appendSignalFrame(base, jogged).segment, 0);
    const missing = frame(2, { capture: undefined }); missing.info.position = null;
    assert.equal(appendSignalFrame(base, missing), base);
});

test('history is bounded and geometry excludes old settings and points outside the local window', () => {
    let series = emptySignalSeries();
    for (let i = 1; i <= 365; i++) series = appendSignalFrame(series, frame(i));
    assert.equal(series.samples.length, 360);
    assert.equal(series.samples[0].time, 6);
    assert.equal(signalGeometry(series, 'p999', 2000, 100).dots.length, 0);
    assert.equal(signalGeometry(series, 'p999', 1000, 100).dots.length, 1);
    const changed = frame(366); changed.capture.channel.exposure_time_ms = 20; changed.capture.position.z_pos_mm = 1.001;
    const next = appendSignalFrame(series, changed);
    assert.equal(signalGeometry(next, 'p999', 1000, 100).dots.length, 1);
});

test('bounded Z jog targets accept exact endpoints and reject invalid steps or overtravel', () => {
    const range = { min: 0, max: 6700 };
    assert.equal(jogTarget(range, 6699, 1, 1), 6700);
    assert.equal(jogTarget(range, 1, -1, 1), 0);
    assert.equal(jogTarget(range, 6700, 1, 1), null);
    assert.equal(jogTarget(range, 0, -1, 0.1), null);
    for (const step of [0, -1, NaN, Infinity]) assert.equal(jogTarget(range, 100, 1, step), null);
    assert.equal(jogTarget(null, 100, 1, 1), null);
    assert.equal(jogTarget(range, null, 1, 1), null);
    assert.equal(jogTarget(range, 100, 2, 1), null);
});
