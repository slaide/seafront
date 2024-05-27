class SiteSelectionCell{
    /**
     * @param {number} row
     * @param {number} col
     * @param {boolean} selected
     */
    constructor(row,col,selected=true){
        this.row=row
        this.col=col
        this.selected=selected
    }
}

/** @type {SiteSelectionCell[][]} */
let grid_mask=_p.manage([])
const updateGridMask=function(){
    let element=document.getElementById("site-selection-centerer")
    if(!element){console.error("element not found");return}
    if(!(element.parentElement)){console.error("element has no parent");return}

    let num_rows=microscope_config.grid.num_y.valueOf()
    let num_cols=microscope_config.grid.num_x.valueOf()

    grid_mask.length=0

    for(let i=0;i<num_rows;i++){
        let row=[]
        for(let j=0;j<num_cols;j++){
            row.push({row:i,col:j,selected:true})
        }
        grid_mask.push(row)
    }

    let max_num_items=Math.max(num_rows,num_cols)
    element.style.setProperty("--item-size",element.parentElement.clientWidth/max_num_items-2+"px")

    element.style.setProperty("--num-cols",num_cols+"")
    element.style.setProperty("--num-rows",num_rows+"")
}
/**
 * 
 * @param {SiteSelectionCell} cell 
 */
function toggleGridItem(cell){
    grid_mask[cell.row][cell.col].selected=!grid_mask[cell.row][cell.col].selected
}