class SiteSelectionCell{
    /**
     * @param {number} row
     * @param {number} col
     * @param {number} plane
     * @param {boolean} selected
     */
    constructor(row,col,plane,selected=true){
        this.row=row
        this.col=col
        this.plane=plane
        this.selected=selected
    }
}

const updateGridMask=function(){
    let element=document.getElementById("site-selection-centerer")
    if(!element){console.error("element not found");return}
    if(!(element.parentElement)){console.error("element has no parent");return}

    let num_rows=microscope_config.grid.num_y.valueOf()
    let num_cols=microscope_config.grid.num_x.valueOf()

    microscope_config.grid.mask.length=0
    
    // async DOM update
    setTimeout(()=>{
        if(!(element.parentElement)){console.error("element has no parent");return}

        for(let i=0;i<num_rows;i++){
            for(let j=0;j<num_cols;j++){
                const new_cell=new SiteSelectionCell(i,j,0,true)
                microscope_config.grid.mask.push(new_cell)
            }
        }

        let max_num_items=Math.max(num_rows,num_cols)
        element.style.setProperty("--item-size",element.parentElement.clientWidth/max_num_items-2+"px")

        element.style.setProperty("--num-cols",num_cols+"")
        element.style.setProperty("--num-rows",num_rows+"")
    },0)
}
/**
 * 
 * @param {SiteSelectionCell} cell 
 */
function toggleGridItem(cell){
    cell.selected=!cell.selected
}