linkPlots

/**
 * @param{InputEvent} event
 */
function setGridViewCols(event){
	if(!(event.target instanceof HTMLInputElement)){return}
    let num_cols=parseInt(event.target.value)

	const grid_view_container=document.getElementById("grid-view-container")
	if(!grid_view_container){return}
		
    grid_view_container.style.setProperty("--num-cols",""+num_cols)
}