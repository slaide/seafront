/**
 * Focus state after the laser autofocus was calibrated with the reference channel in focus.
 * Channel Z offsets are displacements from the reference channel's focus, so the reference
 * channel's own offset is zero by definition. A leftover value would move that channel out of
 * focus whenever imaging goes through autofocus (snap all channels, acquisition).
 * @param {AcquisitionChannelConfig} referenceChannel its z_offset_um is reset in place
 * @param {Position} position stage position at calibration
 */
export function commitFocusReference(referenceChannel, position) {
    referenceChannel.z_offset_um = 0;
    return {
        focusReferenceZ: position.z_pos_mm,
        focusReferenceXY: { x: position.x_pos_mm, y: position.y_pos_mm },
        /** @type {Set<string>} */
        focusConfirmedChannels: new Set([referenceChannel.handle]),
        /** @type {number|null} */
        focusMeasuredOffsetUM: null,
        /** @type {LaserAutofocusMeasureDisplacementResponse|null} */
        laserAutofocusMeasuredOffset: null,
    };
}

/**
 * Focus state for a page that loads with the laser autofocus already calibrated. The server
 * records the reference channel and the calibration position with the calibration, so the
 * reference survives a reload. Null when there is no calibration, or it does not name a
 * channel of this microscope (one made before the channel was recorded).
 * @param {MachineConfigItem[]} machineConfig
 * @param {AcquisitionChannelConfig[]} channels
 */
export function restoreFocusReference(machineConfig, channels) {
    const value = (/** @type {string} */ name) =>
        machineConfig.find(item => item.handle === `laser.autofocus.calibration.${name}`)?.value;
    if (value('is_calibrated') !== 'yes') return null;
    const handle = value('ref_channel');
    if (!handle || !channels.some(channel => channel.handle === handle)) return null;
    return {
        /** @type {string|null} */
        focusReferenceChannelHandle: String(handle),
        focusReferenceZ: Number(value('ref_z_mm')),
        focusReferenceXY: { x: Number(value('ref_x_mm')), y: Number(value('ref_y_mm')) },
        /** @type {Set<string>} */
        focusConfirmedChannels: new Set([String(handle)]),
    };
}
