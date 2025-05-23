"use strict";

/**
 * r: row index
 * c: col index
 * w: well width (y size) in mm
 * h: well height (x size) in mm
 * pxm: x coordinate on plate in mm
 * pym: y coordinate on plate in mm
 * @typedef {{
 *      r: number,
 *      c: number,
 *      text: string,
 *      w: number,
 *      h: number,
 *      pxm: number,
 *      pym: number
 * }} PhysicalWellItemInformation
 */
/** @type {PhysicalWellItemInformation[]} */
let well_list = _p.manage([])
function generateWellOverviewWells() {
    // clear well_list
    well_list.length = 0

    const plate = microscope_config.wellplate_type

    // async DOM update
    setTimeout(() => {
        /** @type {PhysicalWellItemInformation[]} */
        let ret = []

        for (let c of range(plate.Num_wells_x)) {
            for (let r of range(plate.Num_wells_y)) {
                // add 1 to row and column to account for headers
                const index = new WellIndex(r + 1, c + 1)

                const well_text = index.name
                ret.push({
                    r: r,
                    c: c,
                    text: well_text,
                    w: plate.Well_size_y_mm,
                    h: plate.Well_size_x_mm,
                    pxm: plate.Offset_A1_x_mm + c * plate.Well_distance_x_mm,
                    pym: plate.Offset_A1_y_mm + r * plate.Well_distance_y_mm
                })
            }
        }

        well_list.push(...ret)
    }, 0)
}

// when page has loaded, generate well overview and register callback to update the view when the selected plate type changes
window.addEventListener("load", function () {
    generateWellOverviewWells()
    _p.registerCallback(_p.getUnmanaged(microscope_config).wellplate_type, generateWellOverviewWells)//, "Model_id")
})
