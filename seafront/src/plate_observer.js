/**
 * @typedef {(object&{
 *      r: number,
 *      c: number,
 *      text: string,
 *      w: number,
 *      h: number,
 *      pxm: number,
 *      pym: number
 * })} PhysicalWellItemInformation
 */
/** @type{PhysicalWellItemInformation[]} */
let well_list=_p.manage([])
function generateWellOverviewWells(){
    // clear well_list
    well_list.length=0

    let plate=WellplateType.fromHandle(microscope_config.wellplate_type)
    if(plate==null){
        return
    }

    // async DOM update
    setTimeout(()=>{
        /** @type{PhysicalWellItemInformation[]} */
        let ret=[]
        for(let c of range(plate.num_cols)){
            for(let r of range(plate.num_rows)){
                ret.push({
                    r: r,
                    c: c,
                    text: String.fromCharCode(65 + r) + String(c + 1),
                    w: plate.well_width_mm,
                    h: plate.well_length_mm,
                    pxm: plate.a1_x_offset_mm + c * plate.well_distance_mm,
                    pym: plate.a1_y_offset_mm + r * plate.well_distance_mm
                })
            }
        }

        well_list.push(...ret)
    },0)    
}

// when page has loaded, generate well overview and register callback to update the view when the selected plate type changes
window.addEventListener("DOMContentLoaded",function(){
    generateWellOverviewWells()
    _p.registerCallback(_p.getUnmanaged(microscope_config),generateWellOverviewWells,"wellplate_type")
})
