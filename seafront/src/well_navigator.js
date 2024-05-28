
class WellIndex{
    /**
     * 
     * @param {number} row
     * @param {number} col
     */
    constructor(row,col){
        this.row=row
        this.col=col
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
/** @type{WellIndex[]} */
const plate_wells=_p.manage([])

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

    let new_plate_wells=[]
    
    for(let i=0;i<num_rows;i++){
        for(let j=0;j<num_cols;j++){
            let new_well = new WellIndex(i,j)
            
            new_plate_wells.push(new_well)
        }
    }

    plate_wells.length=0
    plate_wells.splice(0,0,...new_plate_wells)
}

/**
 * 
 * @param {WellIndex} item 
 */
function clickWell(item){
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
