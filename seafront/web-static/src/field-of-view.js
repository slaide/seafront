// Match MAIN_CAMERA_PIXEL_PITCH_UM in server/protocol.py: the calibrated 20x field is
// 0.3 µm/px (3000 px covers 0.9 mm). This is the pitch projected through the tube optics.
const MAIN_CAMERA_PIXEL_PITCH_UM = 6;

/**
 * Planned image footprint from objective option metadata and the calibrated pixel pitch.
 * Use configured crop dimensions, never preview/downsample size.
 * @param {{handle: string, value: unknown, options?: {handle: string, info?: unknown}[]|null}[]|null|undefined} config
 * @returns {{fovx: number, fovy: number}|null}
 */
export function configuredFieldOfView(config) {
    const value = (/** @type {string} */ handle) => config?.find(item => item.handle === handle)?.value;
    const objective = config?.find(item => item.handle === 'camera.main.objective');
    const info = objective?.options?.find(option => option.handle === objective.value)?.info;
    const magnification = info && typeof info === 'object' && 'magnification' in info
        && typeof info.magnification === 'number' ? info.magnification : NaN;
    const width = Number(value('camera.main.image.width_px'));
    const height = Number(value('camera.main.image.height_px'));
    if (![magnification, width, height].every(v => Number.isFinite(v) && v > 0)) return null;
    const mmPerPixel = MAIN_CAMERA_PIXEL_PITCH_UM / magnification / 1000;
    return { fovx: width * mmPerPixel, fovy: height * mmPerPixel };
}
