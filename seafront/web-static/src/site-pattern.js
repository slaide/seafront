/**
 * Fit the well boundary and every camera footprint into a square preview.
 * Site centers follow the acquisition protocol: a centered grid, row zero at the top.
 * @param {{Well_size_x_mm: number, Well_size_y_mm: number, Well_edge_radius_mm: number}} plate
 * @param {{num_x: number, num_y: number, delta_x_mm: number, delta_y_mm: number, mask: {row: number, col: number, selected: boolean}[]}} grid
 * @param {{fovx: number, fovy: number}} fov
 */
export function sitePatternGeometry(plate, grid, fov) {
    const values = [plate.Well_size_x_mm, plate.Well_size_y_mm, fov.fovx, fov.fovy,
        grid.num_x, grid.num_y, grid.delta_x_mm, grid.delta_y_mm];
    if (!values.every(Number.isFinite) || values.slice(0, 6).some(v => v <= 0)) return null;
    const extentX = Math.max(plate.Well_size_x_mm, (grid.num_x - 1) * grid.delta_x_mm + fov.fovx);
    const extentY = Math.max(plate.Well_size_y_mm, (grid.num_y - 1) * grid.delta_y_mm + fov.fovy);
    const scale = 268 / Math.max(extentX, extentY);
    return {
        well: { x: 160 - plate.Well_size_x_mm * scale / 2, y: 160 - plate.Well_size_y_mm * scale / 2,
            width: plate.Well_size_x_mm * scale, height: plate.Well_size_y_mm * scale,
            radius: Math.max(0, plate.Well_edge_radius_mm || 0) * scale },
        sites: grid.mask.map((site, index) => {
            const cx = 160 + (site.col - (grid.num_x - 1) / 2) * grid.delta_x_mm * scale;
            const cy = 160 + (site.row - (grid.num_y - 1) / 2) * grid.delta_y_mm * scale;
            return { index, row: site.row, col: site.col, selected: site.selected, cx, cy,
                x: cx - fov.fovx * scale / 2, y: cy - fov.fovy * scale / 2,
                width: fov.fovx * scale, height: fov.fovy * scale };
        }),
    };
}

/** @param {ReturnType<typeof sitePatternGeometry>} geometry */
export function sitePatternSvg(geometry) {
    if (!geometry) return '';
    const { well, sites } = geometry;
    const label = (/** @type {typeof sites[number]} */ site) => `Site ${site.col + 1}, ${site.row + 1}`;
    return `<rect class="site-well-boundary" x="${well.x}" y="${well.y}" width="${well.width}" height="${well.height}" rx="${well.radius}" />
        <path class="site-center-guide" d="M160 16V304 M16 160H304" />` +
        sites.map(site => `<rect data-site="${site.index}" class="site-footprint ${site.selected ? 'is-selected' : ''}" x="${site.x}" y="${site.y}" width="${site.width}" height="${site.height}"><title>${label(site)} · ${site.selected ? 'selected' : 'excluded'}</title></rect>`).join('') +
        // Draw all center targets after the footprints so overlapping tiles remain selectable.
        sites.map(site => `<g data-site="${site.index}" class="site-center ${site.selected ? 'is-selected' : ''}" tabindex="0" role="button" aria-label="${label(site)}" aria-pressed="${site.selected}"><title>${label(site)} · click to ${site.selected ? 'exclude' : 'select'}</title><circle class="site-hit-target" cx="${site.cx}" cy="${site.cy}" r="10" /><circle cx="${site.cx}" cy="${site.cy}" r="4" /></g>`).join('');
}
