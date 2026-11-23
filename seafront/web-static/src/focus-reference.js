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
