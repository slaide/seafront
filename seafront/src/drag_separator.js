/** @type {{drag_element:null|HTMLElement}} */
let dragInfo={
    drag_element:null,
}
/**
 * @brief event handler to initialize drag
 * @param {MouseEvent} event 
 */
const registerdrag=function(event){
    if(!(event.currentTarget instanceof HTMLElement)){throw new Error("event.currentTarget is not an HTMLElement")}
    dragInfo.drag_element=event.currentTarget
    document.addEventListener("mousemove",dragmouse)
    // on mouse up or mouse leave, remove the event listener
    document.addEventListener("mouseup",unregisterdrag)
    document.addEventListener("mouseleave",unregisterdrag)
}
/**
 * @brief event handler to stop drag
 * @param {MouseEvent} event 
 */
const unregisterdrag=function(event){
    document.removeEventListener("mousemove",dragmouse)
    document.removeEventListener("mouseup",unregisterdrag)
    document.removeEventListener("mouseleave",unregisterdrag)
    this.dragx_start=null;
}
/**
 * 
 * @param {MouseEvent} event 
 */
const dragmouse=function(event){
    let newx=event.clientX
    if(this.dragx_start==null){
        this.dragx_start=newx
    }
    if(!dragInfo.drag_element){throw new Error("drag element is null")}
    if(!(dragInfo.drag_element.parentNode instanceof HTMLElement)){throw new Error("drag element parent is null")}

    const left_frac=newx/window.innerWidth
    dragInfo.drag_element.parentNode.style.setProperty("--leftCol",left_frac+"fr")
    dragInfo.drag_element.parentNode.style.setProperty("--rightCol",(1-left_frac)+"fr")
}