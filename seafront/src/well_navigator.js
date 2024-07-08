
class WellIndex{
    /**
     * 
     * @param {number} row
     * @param {number} col
     * @param {boolean} selected
     */
    constructor(row,col,selected=true){
        this.row=row
        this.col=col
        this.selected=selected
    }
    get isHeader(){
        return this.row==0 || this.col==0
    }
    get row_name(){
        let row_name=String.fromCharCode("A".charCodeAt(0)+this.row-1)
        if(this.row>=27){
            row_name=String.fromCharCode("a".charCodeAt(0)+this.row-27)
        }
        return row_name
    }
    get col_name(){
        let col_name=this.col.toString()
        if(col_name.length<2){
            col_name="0"+col_name
        }

        return col_name
    }
    /**
     * get name of the well, e.g. A01, B12
     * @returns {string}
     */
    get name(){
        return this.row_name+this.col_name
    }
    /**
     * get the name of the well, only if it is a header (otherwise returns empty string)
     * @returns {string}
     */
    get label(){
        // if row is zero, then it is a column header
        if(this.row==0 && this.col>0){
            return this.col_name
        }
        // if column is zero, then the cell is a row header
        if(this.col==0 && this.row>0){
            return this.row_name
        }
        // if row and column are both zero, then it is the top left corner, which is empty
        if(this.row==0 && this.col==0){
            return ""
        }
        return ""
    }

    /** @type{Set<WellIndex>} */
    static forbidden_wells=new Set()

    get forbidden(){
        return WellIndex.forbidden_wells.has(this)
    }
}

/**
 * called when the selected well plate is changed or initially set
 */
function initwellnavigator(){
    let element=document.getElementById("well-navigator-container")
    if(!element){console.error("element not found");return}

    let plate_type=WellplateType.fromHandle(microscope_config.wellplate_type)
    if(!plate_type){console.error("wellplate type not found");return}
    
    let num_cols=plate_type.Num_wells_x
    let num_rows=plate_type.Num_wells_y

    // add 1 for headers
    num_cols+=1
    num_rows+=1

    element.style.setProperty("--num-cols",num_cols+"");
    element.style.setProperty("--num-rows",num_rows+"");

    microscope_config.plate_wells.length=0
    WellIndex.forbidden_wells.clear()

    const plate_type_forbidden_wells=new Set()

    const forbidden_wells_str=machine_defaults.find(v=>v.handle=="forbidden_wells")
    if(!forbidden_wells_str){
        console.error("forbidden_wells not found")
    }else{
        const plate_type_forbidden_wells_str=forbidden_wells_str.value.toString().split(";").filter(s=>s.length>0).map(s=>s.split(":")).find(v=>parseInt(v[0])==plate_type.num_wells)
        if(!plate_type_forbidden_wells_str)throw new Error("forbidden_wells not found for plate type "+plate_type.num_wells)

        plate_type_forbidden_wells_str[1].split(",").forEach(w_str=>plate_type_forbidden_wells.add(w_str))
    }

    let new_plate_wells=[]
    
    for(let i=0;i<num_rows;i++){
        for(let j=0;j<num_cols;j++){
            let new_well = new WellIndex(i,j,false)
            
            new_plate_wells.push(new_well)

            if(plate_type_forbidden_wells.has(new_well.name)){
                WellIndex.forbidden_wells.add(new_well)
            }
        }
    }

    microscope_config.plate_wells.push(...new_plate_wells)
}

/** @type{object&{
 *      start: WellIndex?,
 *      end: WellIndex?,
 *      prev_state: Map< WellIndex, boolean>?
 * }} */
let drag_info={
    start:null,
    end:null,
    prev_state:null,
}
function _wellPointer_update(){
    if(!(drag_info.start && drag_info.end && drag_info.prev_state))return

    let start_row=drag_info.start.row
    let start_col=drag_info.start.col
    let end_row=drag_info.end.row
    let end_col=drag_info.end.col

    let start_row_index=Math.min(start_row,end_row)
    let end_row_index=Math.max(start_row,end_row)

    let start_col_index=Math.min(start_col,end_col)
    let end_col_index=Math.max(start_col,end_col)

    // state to set items to is opposite of state of first item
    let set_state=!drag_info.prev_state.get(drag_info.start)

    for(let i=0;i<microscope_config.plate_wells.length;i++){
        let well=microscope_config.plate_wells[i]

        // skip headers
        if(well.isHeader)continue

        if(!drag_info.prev_state.has(well)){
            drag_info.prev_state.set(well,well.selected)
        }

        // if the well is in the selected range, then it should be selected, otherwise check prev_state and apply that
        if(
            well.row>=start_row_index && well.row<=end_row_index
            && well.col>=start_col_index && well.col<=end_col_index
        ){
            well.selected=set_state
        }else{
            let prev_state=drag_info.prev_state.get(well)
            if(prev_state==undefined)continue
            well.selected=prev_state
        }
    }
}

function wellPointerCancel(){
    if(!(drag_info.start && drag_info.end && drag_info.prev_state))return

    // restore previous state
    for(let i=0;i<microscope_config.plate_wells.length;i++){
        let well=microscope_config.plate_wells[i]

        let prev_well_state=drag_info.prev_state.get(well)
        if(prev_well_state==undefined)continue

        well.selected=prev_well_state
    }

    drag_info.start=null
    drag_info.end=null
    drag_info.prev_state=new Map()
}

/**
 * 
 * @param {PointerEvent} event
 * @param {WellIndex} item 
 */
function wellPointerDown(event,item){
    event.preventDefault()
    drag_info.start=item
    drag_info.end=item
    drag_info.prev_state=new Map()

    drag_info.prev_state.set(item,item.selected)

    _wellPointer_update()
}
/**
 * 
 * @param {PointerEvent} event
 * @param {WellIndex} item 
 */
function wellPointerUp(event,item){
    if(!drag_info.start)return
    event.preventDefault()

    drag_info.start=null
    drag_info.end=null
    drag_info.prev_state=new Map()
}

/**
 * 
 * @param {PointerEvent} event
 * @param {WellIndex} item 
 */
function wellPointerUpdate(event,item){
    if(!drag_info.start)return
    event.preventDefault()

    drag_info.end=item
    
    _wellPointer_update()
}

/**
 * 
 * @param {WellIndex} item 
 */
function dblclickWell(item){
    if(item.col==0 || item.row==0)return;
    //if(item.forbidden)return;

    progress_indicator.run("Moving to well "+item.name)

    const data={
        plate_type: microscope_config.wellplate_type,
        well_name: item.name,
    }

    let xhr=null
    xhr=new XHR(true)
        .onload(function(xhr){
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("failed to move to well",response,item)
                return
            }
        })
        .onerror(function(){
            progress_indicator.stop()
            
            console.error("failed to move to well",item)
        })
        .send("/api/action/move_to_well",data,"POST")
}
