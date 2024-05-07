
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

    let plate_type=microscope_config.wellplate_type.valueOf()
    
    let num_cols=0
    let num_rows=0

    // if plate type ends in 96, then 8x12
    if(plate_type.endsWith("96")){
        num_cols=12
        num_rows=8
    }else if(plate_type.endsWith("384")){
        num_cols=24
        num_rows=16
    }

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

    plate_wells.splice(0,plate_wells.length,...new_plate_wells)
}