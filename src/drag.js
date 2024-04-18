/**
 * @brief event handler to initialize drag
 * @param {Info} info 
 */
p.registerdrag=function(info){
    this.drag_element=info.target
    document.addEventListener("mousemove",p.dragmouse);
    // on mouse up or mouse leave, remove the event listener
    document.addEventListener("mouseup",p.unregisterdrag);
    document.addEventListener("mouseleave",p.unregisterdrag);
}
/**
 * @brief event handler to stop drag
 * @param {MouseEvent} event 
 */
p.unregisterdrag=function(event){
    document.removeEventListener("mousemove",p.dragmouse);
    document.removeEventListener("mouseup",p.unregisterdrag);
    document.removeEventListener("mouseleave",p.unregisterdrag);
    this.dragx_start=null;
}
/**
 * 
 * @param {MouseEvent} event 
 */
p.dragmouse=function(event){
    let newx=event.clientX;
    if(this.dragx_start==null){
        this.dragx_start=newx;
    }
    this.drag_element.parentNode.style.setProperty("--left-fraction",newx/window.innerWidth);
}