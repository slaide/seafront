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
