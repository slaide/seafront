class PlotData{
    /**
     * container for plot data (mainly axis limits)
     * @param {number} x_min 
     * @param {number} x_max 
     * @param {number} y_min 
     * @param {number} y_max 
     */
    constructor(x_min,x_max,y_min,y_max){
        this.x_min=x_min
        this.x_max=x_max
        this.y_min=y_min
        this.y_max=y_max
    }

    getDeltaX(){
        return this.x_max-this.x_min
    }
    getDeltaY(){
        return this.y_max-this.y_min
    }
}
/** @type {Map<HTMLElement,PlotData>} */
let elementPlotData=new Map()
/**
 * retrieve plot data (mainly axis limits) for a given plot element
 * @param {HTMLElement} element 
 * @returns {PlotData}
 */
function getElementPlotData(element){
    if(!elementPlotData.has(element)){
        elementPlotData.set(element,new PlotData(0,0,0,0))
    }
    let ret=elementPlotData.get(element)
    if(!ret){throw new Error("unreachable")}
    return ret
}
/**
 * update plot display given current state
 * @param {HTMLElement} plot 
 */
const plot_update=function(plot){
    let plot_data=getElementPlotData(plot)
    let x_range = plot_data.x_max - plot_data.x_min;
    let y_range = plot_data.y_max - plot_data.y_min;

    let plot_x_size=plot.clientWidth;
    let plot_y_size=plot.clientHeight;

    let scale_x=(plot_x_size/x_range);
    let scale_y=(plot_y_size/y_range);

    for(let child of plot.children){
        if(!(child instanceof HTMLElement)){continue}
        let child_plot_data=getElementPlotData(child)
        let c_width=child_plot_data.getDeltaX()
        let c_height=child_plot_data.getDeltaY()

        child.style.setProperty("--scale-x",scale_x+"")
        child.style.setProperty("--scale-y",scale_y+"")

        child.style.setProperty("--left",(child_plot_data.x_min-plot_data.x_min)*scale_x+"px")
        child.style.setProperty("--bottom",(child_plot_data.y_min-plot_data.y_min)*scale_y+"px")
    }
}
let zoom_speed=0.01;
let invert_scroll=true;

/**
 * handle zoom (scroll/wheel) event on plot
 * @param {WheelEvent} event 
 */
function plot_zoom(event){
    event.preventDefault();

    let plot = event.currentTarget
    if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

    let zoom_delta=event.deltaY;
    if(Math.abs(zoom_delta)>5){
        zoom_delta=Math.sign(zoom_delta)*5;
    }

    // scrolling up naturally zooms out
    let zoom=zoom_speed*zoom_delta;
    if(invert_scroll){
        zoom*=-1;
    }
    
    let plot_data=getElementPlotData(plot)
    let x_range = plot_data.getDeltaX()
    let y_range = plot_data.getDeltaY()

    plot_data.x_min += x_range*zoom
    plot_data.x_max -= x_range*zoom
    plot_data.y_min += y_range*zoom
    plot_data.y_max -= y_range*zoom

    plot_update(plot)
}
/** @type {Map<HTMLElement,{in_progress:boolean,x_start:number,y_start:number}>} */
const plotDragInfo=new Map()
/**
 * init plot dragging (handles mousedown)
 * @param {MouseEvent} event 
 */
function plot_drag_start(event){
    if(!(event.currentTarget instanceof HTMLElement)){throw new Error("plot is not an html element")}
    plotDragInfo.set(event.currentTarget,{in_progress:true,x_start:event.clientX,y_start:event.clientY})
}
/**
 * process mouse move while dragging plot
 * @param {MouseEvent} event 
 */
function plot_drag_move(event){
    let plot = event.currentTarget
    if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

    let plotDragData=plotDragInfo.get(plot)
    if(plotDragData && plotDragData.in_progress){
        event.preventDefault()

        let plot_data=getElementPlotData(plot)
        let x_range = plot_data.getDeltaX()
        let y_range = plot_data.getDeltaY()

        let plot_x_size=plot.clientWidth
        let plot_y_size=plot.clientHeight

        let scale_x=(plot_x_size/x_range)
        let scale_y=(plot_y_size/y_range)

        let x_delta = event.clientX-plotDragData.x_start
        let y_delta = event.clientY-plotDragData.y_start

        plot_data.x_min -= x_delta/scale_x
        plot_data.x_max -= x_delta/scale_x
        plot_data.y_min += y_delta/scale_y
        plot_data.y_max += y_delta/scale_y

        plotDragData.x_start=event.clientX
        plotDragData.y_start=event.clientY

        plot_update(plot)
    }
}

/**
 * end dragging (handles mouseup and mouseleave)
 * @param {MouseEvent} event 
 */
function plot_drag_end(event){
    let plot = event.currentTarget
    if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

    let plotDragData=plotDragInfo.get(plot)
    if(!plotDragData){return}

    plotDragData.in_progress=false
    plotDragInfo.delete(plot)
}
/**
 * init element as plot display
 * @param {HTMLElement} plot 
 */
function plot_init(plot){
    for(let child of plot.children){
        if(!(child instanceof HTMLElement)){continue}
        let child_plot_data=getElementPlotData(child)
        child_plot_data.x_min=0
        child_plot_data.x_max=child.clientWidth
        child_plot_data.y_min=0
        child_plot_data.y_max=child.clientHeight

        child.style.setProperty("--width",child.clientWidth+"px");
        child.style.setProperty("--height",child.clientHeight+"px");
        child.classList.add("plot-img")
    }

    plot_fit(plot)
}
/**
 * fit plot to display all children
 * @param {Event|HTMLElement} plot 
 */
function plot_fit(plot){
    let plot_element=(plot instanceof HTMLElement)?plot:plot.currentTarget
    if(!(plot_element instanceof HTMLElement)){throw new Error("plot element is not an html element")}
    // adjust plot to fit the children

    let x_min=Infinity
    let x_max=-Infinity
    let y_min=Infinity
    let y_max=-Infinity

    for(let child of plot_element.children){
        if(!(child instanceof HTMLElement)){continue}
        let child_plot_data=getElementPlotData(child)
        x_min=Math.min(x_min,child_plot_data.x_min)
        x_max=Math.max(x_max,child_plot_data.x_max)
        y_min=Math.min(y_min,child_plot_data.y_min)
        y_max=Math.max(y_max,child_plot_data.y_max)
    }

    let plot_data=getElementPlotData(plot_element)
    plot_data.x_min=x_min
    plot_data.x_max=x_max
    plot_data.y_min=y_min
    plot_data.y_max=y_max

    plot_update(plot_element)
}
