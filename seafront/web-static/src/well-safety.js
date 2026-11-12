/**
 * Check the physical well center, using the same coordinates as MoveToWell on the server.
 * Plate display coordinates flip Y; safety checks must use the physical row index instead.
 * @param {Wellplate} plate
 * @param {{row: number, col: number}} well
 * @param {ForbiddenArea[]} areas
 * @returns {ForbiddenArea|undefined}
 */
export function wellForbiddenArea(plate, well, areas) {
    if (well.row < 0 || well.col < 0) return undefined;
    const x = plate.Offset_A1_x_mm + well.col * plate.Well_distance_x_mm + plate.Well_size_x_mm / 2;
    const y = plate.Offset_A1_y_mm + well.row * plate.Well_distance_y_mm + plate.Well_size_y_mm / 2;
    return areas.find(area => x >= area.min_x_mm && x <= area.max_x_mm &&
        y >= area.min_y_mm && y <= area.max_y_mm);
}
