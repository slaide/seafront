"use strict";

/** @typedef {'p99'|'p999'|'p9999'|'max'} SignalMetric */
/** @typedef {{p99:number,p999:number,p9999:number,max:number,saturated:number,pixels:number}} SignalStats */
/** @typedef {{id:string,time:number,z:number,key:string,segment:number,stats:SignalStats}} SignalSample */
/** @typedef {{samples:SignalSample[],segment:number,lastId:string,lastTime:number}} SignalSeries */

/** Exact nearest-rank percentiles of the transmitted preview, in stored integer units.
 * @param {ArrayBuffer} buffer
 * @param {number} bits
 * @param {number} cameraBits
 * @returns {SignalStats}
 */
export function imageSignalStats(buffer, bits, cameraBits) {
    if (bits !== 8 && bits !== 16) throw new Error(`Unsupported preview depth: ${bits}`);
    const pixels = bits === 16 ? new Uint16Array(buffer) : new Uint8Array(buffer);
    if (!pixels.length) throw new Error("Empty preview");
    const counts = new Uint32Array(2 ** bits);
    let max = 0;
    for (const pixel of pixels) {
        counts[pixel]++;
        if (pixel > max) max = pixel;
    }
    const ranks = [0.99, 0.999, 0.9999].map(p => Math.ceil(p * pixels.length));
    const values = [0, 0, 0];
    let total = 0, rank = 0;
    for (let value = 0; value <= max && rank < ranks.length; value++) {
        total += counts[value];
        while (rank < ranks.length && total >= ranks[rank]) values[rank++] = value;
    }
    const significant = Number.isInteger(cameraBits) && cameraBits > 0 ? Math.min(bits, cameraBits) : bits;
    const nativeCeiling = 2 ** significant - 1;
    const paddedCeiling = nativeCeiling * 2 ** (bits - significant);
    // Camera samples may be native, bit-shifted, or expanded to the container's full scale.
    // Deduplicate the ceilings when camera and container bit depths match.
    const ceilings = new Set([nativeCeiling, paddedCeiling, 2 ** bits - 1]);
    const clipped = [...ceilings].reduce((sum, ceiling) => sum + counts[ceiling], 0);
    return { p99: values[0], p999: values[1], p9999: values[2], max, saturated: clipped / pixels.length, pixels: pixels.length };
}

/** @returns {SignalSeries} */
export function emptySignalSeries() {
    return { samples: [], segment: 0, lastId: "", lastTime: -Infinity };
}

/** @param {AcquisitionChannelConfig} channel */
export function imagingKey(channel) {
    return JSON.stringify([channel.exposure_time_ms, channel.analog_gain, channel.illum_perc, channel.filter_handle ?? null]);
}

/** Append in capture order, breaking lines when capture settings or XY change.
 * @param {SignalSeries} series
 * @param {CachedChannelImage} image
 * @returns {SignalSeries}
 */
export function appendSignalFrame(series, image) {
    const capture = image.capture;
    const time = capture?.timestamp ?? image.info.timestamp;
    const id = capture?.image_id || image.info.image_id || String(time);
    if (id === series.lastId || time <= series.lastTime) return series;
    const position = capture?.position ?? image.info.position?.position;
    if (!position || ![position.x_pos_mm, position.y_pos_mm, position.z_pos_mm, time].every(Number.isFinite)) return series;
    const key = JSON.stringify([
        imagingKey(capture?.channel ?? image.info.channel),
        Math.round(position.x_pos_mm * 1000), Math.round(position.y_pos_mm * 1000),
        image.width, image.height, image.bit_depth, image.camera_bit_depth,
        capture?.pixel_format ?? "",
    ]);
    const previous = series.samples.at(-1);
    const segment = series.segment + (previous && previous.key !== key ? 1 : 0);
    const sample = { id, time, z: position.z_pos_mm * 1000, key, segment,
        stats: imageSignalStats(image.data, image.bit_depth, image.camera_bit_depth) };
    return { samples: [...series.samples.slice(-359), sample], segment, lastId: id, lastTime: time };
}

/** @type {WeakMap<SignalSeries, Map<string, ReturnType<typeof buildSignalGeometry>>>} */
const geometryCache = new WeakMap();

/** @param {SignalSeries|undefined} series @param {SignalMetric} metric @param {number} center @param {number} span */
export function signalGeometry(series, metric, center, span) {
    if (!series) return buildSignalGeometry(series, metric, center, span);
    let cache = geometryCache.get(series);
    if (!cache) { cache = new Map(); geometryCache.set(series, cache); }
    const key = `${metric}:${center}:${span}`;
    let geometry = cache.get(key);
    if (!geometry) {
        geometry = buildSignalGeometry(series, metric, center, span);
        if (cache.size >= 12) cache.clear();
        cache.set(key, geometry);
    }
    return geometry;
}

/** @param {{min:number,max:number}|null} range @param {number|null} z @param {number} direction @param {number} step */
export function jogTarget(range, z, direction, step) {
    if (!range || z == null || !Number.isFinite(z) || !Number.isFinite(step) || step <= 0 ||
        (direction !== -1 && direction !== 1)) return null;
    const target = z + direction * step;
    return target >= range.min - 1e-8 && target <= range.max + 1e-8 ? target : null;
}

/** Use the most recent contiguous settings/XY segment and median repeated Z readings.
 * @param {SignalSeries|undefined} series
 * @param {SignalMetric} metric
 * @param {number} center
 * @param {number} span
 */
function buildSignalGeometry(series, metric, center, span) {
    const start = center - span / 2, end = center + span / 2;
    const samples = (series?.samples ?? []).filter(p => p.segment === series?.segment && p.z >= start && p.z <= end);
    /** @type {Map<number, number[]>} */
    const bins = new Map();
    for (const sample of samples) {
        const z = Math.round(sample.z * 10) / 10;
        const values = bins.get(z) ?? [];
        values.push(sample.stats[metric]);
        bins.set(z, values.slice(-5));
    }
    const points = [...bins.entries()].map(([z, values]) => {
        values.sort((a, b) => a - b);
        return { z, value: values[Math.floor(values.length / 2)] };
    }).sort((a, b) => a.z - b.z);
    const observedLow = points.length ? Math.min(...points.map(p => p.value)) : 0;
    const observedHigh = points.length ? Math.max(...points.map(p => p.value)) : 1;
    const padding = Math.max((observedHigh - observedLow) * 0.1, observedHigh * 0.01, 1);
    const low = Math.max(0, observedLow - padding), high = observedHigh + padding;
    const x = (/** @type {number} */ z) => 10 + (z - start) / span * 210;
    const y = (/** @type {number} */ value) => 83 - (value - low) / (high - low) * 70;
    return {
        path: points.map((p, i) => `${i ? 'L' : 'M'}${x(p.z).toFixed(2)},${y(p.value).toFixed(2)}`).join(' '),
        dots: points.map(p => ({ x: x(p.z), y: y(p.value) })),
        dotsPath: points.map(p => `M${x(p.z).toFixed(2)},${y(p.value).toFixed(2)}h0.01`).join(' '),
        start, end, high, low,
    };
}
