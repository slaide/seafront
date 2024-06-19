
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
    get name(){
        let row_name=String.fromCharCode(64+this.row)
        let col_name=this.col
        return row_name+col_name
    }
    get label(){
        // if row is zero, then it is a column header
        if(this.row==0 && this.col>0){
            return this.col
        }
        // if column is zero, then the cell is a row header
        if(this.col==0 && this.row>0){
            return String.fromCharCode(64+this.row)
        }
        // if row and column are both zero, then it is the top left corner, which is empty
        if(this.row==0 && this.col==0){
            return ""
        }
        return ""
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
    
    let num_cols=plate_type.num_cols
    let num_rows=plate_type.num_rows

    // add 1 for headers
    num_cols+=1
    num_rows+=1

    element.style.setProperty("--num-cols",num_cols+"");
    element.style.setProperty("--num-rows",num_rows+"");

    microscope_config.plate_wells.length=0

    let new_plate_wells=[]
    
    for(let i=0;i<num_rows;i++){
        for(let j=0;j<num_cols;j++){
            let new_well = new WellIndex(i,j,false)
            
            // bug in p2.js: if the object is not managed, the container will
            //      trigger change updates on every member access, and receive
            //      an additional element that is undefined
            new_plate_wells.push(_p.manage(new_well))
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

    let xhr=null
    xhr=new XHR(true)
        .onload(function(xhr){
            console.log("moved to well",item)
        })
        .onerror(function(){
            console.error("failed to move to well",item)
        })
        .send("/api/action/move_to_well",{well_name:item.name},"POST")
}
