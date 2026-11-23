/**
 * Resolve a pointer to the nearest well center, including the gutters between cells.
 * Once dragging, clamp to the plate so release outside the grid still finishes the range.
 * @param {number} x
 * @param {number} y
 * @param {{left:number, top:number, width:number, height:number}} first
 * @param {{left:number, top:number, width:number, height:number}} last
 * @param {number} columns
 * @param {number} rows
 * @param {boolean} clamp
 */
export function wellGridHit(x, y, first, last, columns, rows, clamp = false) {
    if (!clamp && (x < first.left || y < first.top ||
        x > last.left + last.width || y > last.top + last.height)) return null;
    const stepX = columns > 1 ? (last.left - first.left) / (columns - 1) : first.width;
    const stepY = rows > 1 ? (last.top - first.top) / (rows - 1) : first.height;
    if (stepX <= 0 || stepY <= 0) return null;
    return {
        col: Math.max(0, Math.min(columns - 1, Math.round((x - first.left - first.width / 2) / stepX))),
        row: Math.max(0, Math.min(rows - 1, Math.round((y - first.top - first.height / 2) / stepY))),
    };
}

/**
 * CSS border-radius for a well cell, from the plate's corner radius. Cells scale with the
 * panel, so the radius is a share of the well's size: CSS takes horizontal percentages of the
 * width and vertical ones of the height, which keeps corners round in non-square wells. A
 * radius of half the well draws a circle, as in the 3D plate view.
 * @param {{Well_size_x_mm: number, Well_size_y_mm: number, Well_edge_radius_mm: number}} plate
 * @returns {string}
 */
export function wellCornerRadius(plate) {
    const radius = Math.max(0, plate.Well_edge_radius_mm || 0);
    const share = (/** @type {number} */ size) =>
        size > 0 ? Number(Math.min(50, radius / size * 100).toFixed(3)) : 0;
    return `${share(plate.Well_size_x_mm)}% / ${share(plate.Well_size_y_mm)}%`;
}
