/**
 * Fit the well boundary and every camera footprint into a square preview.
 * Site centers follow the acquisition protocol: a centered grid, row zero at the top.
 * `margin` is the well outline shrunk by the edge margin, or null without a margin.
 * @param {{Well_size_x_mm: number, Well_size_y_mm: number, Well_edge_radius_mm: number}} plate
 * @param {{num_x: number, num_y: number, delta_x_mm: number, delta_y_mm: number, mask: {row: number, col: number, selected: boolean}[]}} grid
 * @param {{fovx: number, fovy: number}} fov
 * @param {number} [margin_mm]
 */
export function sitePatternGeometry(plate, grid, fov, margin_mm = 0) {
    const values = [plate.Well_size_x_mm, plate.Well_size_y_mm, fov.fovx, fov.fovy,
        grid.num_x, grid.num_y, grid.delta_x_mm, grid.delta_y_mm];
    if (!values.every(Number.isFinite) || values.slice(0, 6).some(v => v <= 0)) return null;
    const extentX = Math.max(plate.Well_size_x_mm, (grid.num_x - 1) * grid.delta_x_mm + fov.fovx);
    const extentY = Math.max(plate.Well_size_y_mm, (grid.num_y - 1) * grid.delta_y_mm + fov.fovy);
    const scale = 268 / Math.max(extentX, extentY);
    const inset = wellInset(plate, margin_mm);
    return {
        well: { x: 160 - plate.Well_size_x_mm * scale / 2, y: 160 - plate.Well_size_y_mm * scale / 2,
            width: plate.Well_size_x_mm * scale, height: plate.Well_size_y_mm * scale,
            radius: Math.max(0, plate.Well_edge_radius_mm || 0) * scale },
        margin: inset.margin > 0 && inset.halfX > 0 && inset.halfY > 0 ? {
            x: 160 - inset.halfX * scale, y: 160 - inset.halfY * scale,
            width: 2 * inset.halfX * scale, height: 2 * inset.halfY * scale, radius: inset.radius * scale,
        } : null,
        sites: grid.mask.map((site, index) => {
            const cx = 160 + (site.col - (grid.num_x - 1) / 2) * grid.delta_x_mm * scale;
            const cy = 160 + (site.row - (grid.num_y - 1) / 2) * grid.delta_y_mm * scale;
            return { index, row: site.row, col: site.col, selected: site.selected, cx, cy,
                x: cx - fov.fovx * scale / 2, y: cy - fov.fovy * scale / 2,
                width: fov.fovx * scale, height: fov.fovy * scale };
        }),
    };
}

/**
 * @typedef {{Well_size_x_mm: number, Well_size_y_mm: number, Well_edge_radius_mm: number}} WellShape
 * @typedef {{num_x: number, num_y: number, delta_x_mm: number, delta_y_mm: number}} SiteLayout
 */

/**
 * The well outline shrunk inward by an edge margin, as half-sizes and corner radius. Shrinking
 * a rounded rectangle keeps it one: the half-sizes and the radius each lose the margin, and
 * a margin wider than the radius leaves a sharp corner.
 * @param {WellShape} plate
 * @param {number} [margin_mm]
 */
function wellInset(plate, margin_mm = 0) {
    const margin = Math.max(0, Number(margin_mm) || 0);
    const halfX = plate.Well_size_x_mm / 2, halfY = plate.Well_size_y_mm / 2;
    const radius = Math.min(Math.max(0, plate.Well_edge_radius_mm || 0), halfX, halfY);
    return { margin, halfX: halfX - margin, halfY: halfY - margin, radius: Math.max(0, radius - margin) };
}

/**
 * Whether a site's center lies on or inside the well outline: a rectangle with corners rounded by
 * Well_edge_radius_mm, so a well whose radius is half its size is round. `margin_mm` moves the
 * outline inward, to keep sites away from the well edge. Without well dimensions every site
 * counts as inside.
 * @param {WellShape} plate
 * @param {SiteLayout} grid
 * @param {{row: number, col: number}} site
 * @param {number} [margin_mm]
 */
export function siteCenterInWell(plate, grid, site, margin_mm = 0) {
    if (!(plate.Well_size_x_mm > 0 && plate.Well_size_y_mm > 0)) return true;
    const { halfX, halfY, radius } = wellInset(plate, margin_mm);
    if (halfX < 0 || halfY < 0) return false;
    // a tolerance keeps sites placed exactly on the outline (e.g. delta = well radius) inside
    const tolerance = 1e-9;
    const dx = Math.abs((site.col - (grid.num_x - 1) / 2) * grid.delta_x_mm);
    const dy = Math.abs((site.row - (grid.num_y - 1) / 2) * grid.delta_y_mm);
    if (dx > halfX + tolerance || dy > halfY + tolerance) return false;
    const cornerX = dx - (halfX - radius), cornerY = dy - (halfY - radius);
    return cornerX <= 0 || cornerY <= 0 || Math.hypot(cornerX, cornerY) <= radius + tolerance;
}

/**
 * The site mask for `grid`, whose `mask` holds the selection made for an earlier layout.
 * A site starts selected when its center lies inside the well, less the edge margin. It keeps
 * the user's choice after that, unless it is new or moved across that outline since `previous`,
 * the layout and margin the selection was made for. When that layout is unknown (null), e.g.
 * for a loaded protocol, every existing site keeps its selection.
 * @param {WellShape} plate
 * @param {SiteLayout & {mask: {row: number, col: number, selected: boolean}[]}} grid
 * @param {{plate: WellShape, grid: SiteLayout, margin_mm?: number}|null} previous
 * @param {number} [margin_mm]
 */
export function siteMask(plate, grid, previous, margin_mm = 0) {
    const before = new Map(grid.mask.map(site => [`${site.col},${site.row}`, site]));
    const mask = [];
    for (let col = 0; col < grid.num_x; col++) {
        for (let row = 0; row < grid.num_y; row++) {
            const inside = siteCenterInWell(plate, grid, { col, row }, margin_mm);
            const old = before.get(`${col},${row}`);
            const keep = old && (!previous ||
                siteCenterInWell(previous.plate, previous.grid, old, previous.margin_mm) === inside);
            mask.push({ col, row, selected: keep ? old.selected : inside });
        }
    }
    return mask;
}

/**
 * Find the closest site center, including gaps between footprints.
 * @param {ReturnType<typeof sitePatternGeometry>} geometry
 * @param {number} x SVG view-box coordinate
 * @param {number} y SVG view-box coordinate
 * @param {boolean} [clamp] Continue a captured drag beyond the pattern bounds.
 */
export function sitePatternHit(geometry, x, y, clamp = false) {
    if (!geometry?.sites.length) return null;
    const sites = geometry.sites;
    if (!clamp && (x < Math.min(...sites.map(s => s.x)) ||
        y < Math.min(...sites.map(s => s.y)) ||
        x > Math.max(...sites.map(s => s.x + s.width)) ||
        y > Math.max(...sites.map(s => s.y + s.height)))) return null;
    return sites.reduce((nearest, site) =>
        Math.hypot(site.cx - x, site.cy - y) < Math.hypot(nearest.cx - x, nearest.cy - y) ? site : nearest);
}

/**
 * @param {{row: number, col: number}} site
 * @param {{from: {row: number, col: number}, to: {row: number, col: number}}} range
 */
export function siteRangeContains(site, range) {
    return site.row >= Math.min(range.from.row, range.to.row) &&
        site.row <= Math.max(range.from.row, range.to.row) &&
        site.col >= Math.min(range.from.col, range.to.col) &&
        site.col <= Math.max(range.from.col, range.to.col);
}

/**
 * @param {ReturnType<typeof sitePatternGeometry>} geometry
 * @param {{from: {row: number, col: number}, to: {row: number, col: number}, selected: boolean}|null} [selection]
 */
export function sitePatternSvg(geometry, selection = null) {
    if (!geometry) return '';
    const { well, margin, sites } = geometry;
    const label = (/** @type {typeof sites[number]} */ site) => `Site ${site.col + 1}, ${site.row + 1}`;
    const preview = (/** @type {typeof sites[number]} */ site) =>
        selection && siteRangeContains(site, selection) ? selection.selected ? 'range-add' : 'range-remove' : '';
    return `<rect class="site-well-boundary" x="${well.x}" y="${well.y}" width="${well.width}" height="${well.height}" rx="${well.radius}" />
        <path class="site-center-guide" d="M160 16V304 M16 160H304" />` +
        (margin ? `<rect class="site-well-margin" x="${margin.x}" y="${margin.y}" width="${margin.width}" height="${margin.height}" rx="${margin.radius}" />` : '') +
        sites.map(site => `<rect data-site="${site.index}" class="site-footprint ${site.selected ? 'is-selected' : ''} ${preview(site)}" x="${site.x}" y="${site.y}" width="${site.width}" height="${site.height}"><title>${label(site)} · ${site.selected ? 'selected' : 'excluded'}</title></rect>`).join('') +
        // Draw all center targets after the footprints so overlapping tiles remain selectable.
        sites.map(site => `<g data-site="${site.index}" class="site-center ${site.selected ? 'is-selected' : ''} ${preview(site)}" tabindex="0" role="button" aria-label="${label(site)}" aria-pressed="${site.selected}"><title>${label(site)} · click or drag to ${site.selected ? 'exclude' : 'select'}</title><circle class="site-hit-target" cx="${site.cx}" cy="${site.cy}" r="10" /><circle cx="${site.cx}" cy="${site.cy}" r="4" />${preview(site) === 'range-remove' ? `<path class="site-removal-mark" d="M${site.cx - 3} ${site.cy}h6" />` : ''}</g>`).join('');
}
