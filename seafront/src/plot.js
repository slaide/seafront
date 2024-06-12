class PlotData{
    /**
     * container for plot data (mainly axis limits)
     * @param {number} x_min 
     * @param {number} x_max 
     * @param {number} y_min 
     * @param {number} y_max 
     * @param {HTMLElement[]} elements elements to share this plot data with
     */
    constructor(x_min,x_max,y_min,y_max,elements=[]){
        this.x_min=x_min
        this.x_max=x_max
        this.y_min=y_min
        this.y_max=y_max
        this.elements=elements
    }

    /**
     * get delta x (x_max-x_min)
     * @returns {number}
     */
    getDeltaX(){
        return this.x_max-this.x_min
    }
    /**
     * get delta y (y_max-y_min)
     * @returns {number}
     */
    getDeltaY(){
        return this.y_max-this.y_min
    }

    /**
     * 
     * @param {HTMLElement} element 
     * @returns {number}
     */
    static getClientWidth(element){
        let width=element.clientWidth
        if(width==0){
            let width_attribute=element.getAttribute("width")
            if(width_attribute){
                width=parseFloat(width_attribute)
            }
        }
        return width
    }
    /**
     * 
     * @param {HTMLElement} element
     * @returns {number}
     */
    static getClientHeight(element){
        let height=element.clientHeight
        if(height==0){
            let height_attribute=element.getAttribute("height")
            if(height_attribute){
                height=parseFloat(height_attribute)
            }
        }
        return height
    }

    /** @type {Map<HTMLElement,PlotData>} */
    static _elementPlotData=new Map()
    /**
     * retrieve plot data (mainly axis limits) for a given plot element
     * @param {HTMLElement} element 
     * @returns {PlotData}
     */
    static getFor(element){
        if(!this._elementPlotData.has(element)){
            let initial_x_min=parseFloat(element.getAttribute("plot-x-min")||"nan")
            let initial_x_max=parseFloat(element.getAttribute("plot-x-max")||"nan")
            let initial_y_min=parseFloat(element.getAttribute("plot-y-min")||"nan")
            let initial_y_max=parseFloat(element.getAttribute("plot-y-max")||"nan")

            // if initial_x_min is nan, set to 0
            if(isNaN(initial_x_min)){
                initial_x_min=0
            }
            // if initial_x_max is nan, set to initial_x_min+child.clientWidth
            if(isNaN(initial_x_max)){
                let element_width=this.getClientWidth(element)
                initial_x_max=initial_x_min+element_width
            }
            // if initial_y_min is nan, set to 0
            if(isNaN(initial_y_min)){
                initial_y_min=0
            }
            // if initial_y_max is nan, set to initial_y_min+child.clientHeight
            if(isNaN(initial_y_max)){
                let element_height=this.getClientHeight(element)
                initial_y_max=initial_y_min+element_height
            }

            this._elementPlotData.set(element,new PlotData(initial_x_min,initial_x_max,initial_y_min,initial_y_max,[element]))
        }

        let ret=this._elementPlotData.get(element)
        if(!ret){throw new Error("unreachable")}

        return ret
    }

    update(){
        for(let element of this.elements){
            plot_update(element)
        }
    }
    /**
     * link this PlotData element instance to a given HTMLElement
     * 
     * handles duplicates (by ignoring them, i.e. will not add the same element twice)
     * @param {HTMLElement} element 
     */
    link(element){
        // override existing plot data
        PlotData._elementPlotData.set(element,this)
        for(let existing_element of this.elements){
            if(existing_element===element){
                return
            }
        }
        // add new element to list of elements associated with this plot data
        this.elements.push(element)
    }
}

/**
 * update plot display given current state
 * @param {HTMLElement} plot 
 */
const plot_update=function(plot){
    let plot_data=PlotData.getFor(plot)

    let x_range = plot_data.x_max - plot_data.x_min;
    let y_range = plot_data.y_max - plot_data.y_min;

    let plot_x_size=PlotData.getClientWidth(plot)
    let plot_y_size=PlotData.getClientHeight(plot)

    let scale_x=(plot_x_size/x_range);
    let scale_y=(plot_y_size/y_range);

    for(let child of plot.children){
        if(!(child instanceof HTMLElement)){continue}
        let child_plot_data=PlotData.getFor(child)
        let c_width=child_plot_data.getDeltaX()
        let c_height=child_plot_data.getDeltaY()

        child.style.setProperty("--scale-x",scale_x+"")
        child.style.setProperty("--scale-y",scale_y+"")

        child.style.setProperty("--left",(child_plot_data.x_min-plot_data.x_min)*scale_x+"px")
        child.style.setProperty("--bottom",(child_plot_data.y_min-plot_data.y_min)*scale_y+"px")
    }
}
let zoom_speed=0.02;
let invert_scroll=true;

/**
 * handle zoom (scroll/wheel) event on plot
 * @param {WheelEvent} event 
 */
function plot_zoom(event){
    event.preventDefault()

    let plot = event.currentTarget
    if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

    let zoom_delta=event.deltaY
    const MAX_ZOOM_DELTA=5
    if(Math.abs(zoom_delta)>MAX_ZOOM_DELTA){
        zoom_delta=Math.sign(zoom_delta)*MAX_ZOOM_DELTA
    }

    // scrolling up naturally zooms out
    let zoom=zoom_speed*zoom_delta
    if(invert_scroll){
        zoom*=-1
    }
    
    let plot_data=PlotData.getFor(plot)
    const x_range = plot_data.getDeltaX()
    const y_range = plot_data.getDeltaY()

    // zoom in on cursor position with the frame
    const rect=plot.getBoundingClientRect()
    let x_frac=(event.clientX-rect.left)/rect.width
    let y_frac=(event.clientY-rect.top)/rect.height

    const zoom_balance_xmin_frac=x_frac
    const zoom_balance_xmax_frac=1-zoom_balance_xmin_frac
    // y is inverted compared to x because html origin is top left
    const zoom_balance_ymax_frac=y_frac
    const zoom_balance_ymin_frac=1-zoom_balance_ymax_frac

    plot_data.x_min += x_range*zoom*zoom_balance_xmin_frac
    plot_data.x_max -= x_range*zoom*zoom_balance_xmax_frac
    plot_data.y_min += y_range*zoom*zoom_balance_ymin_frac
    plot_data.y_max -= y_range*zoom*zoom_balance_ymax_frac

    plot_data.update()
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

        let plot_data=PlotData.getFor(plot)
        let x_range = plot_data.getDeltaX()
        let y_range = plot_data.getDeltaY()

        let plot_x_size=PlotData.getClientWidth(plot)
        let plot_y_size=PlotData.getClientHeight(plot)

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

        plot_data.update()
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
        // ensure there is plot data for each child
        PlotData.getFor(child)

        let child_width=PlotData.getClientWidth(child)
        let child_height=PlotData.getClientHeight(child)

        child.style.setProperty("--width",child_width+"px");
        child.style.setProperty("--height",child_height+"px");
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

        let child_plot_data=PlotData.getFor(child)

        x_min=Math.min(x_min,child_plot_data.x_min)
        x_max=Math.max(x_max,child_plot_data.x_max)
        y_min=Math.min(y_min,child_plot_data.y_min)
        y_max=Math.max(y_max,child_plot_data.y_max)
    }

    let plot_data=PlotData.getFor(plot_element)
    plot_data.x_min=x_min
    plot_data.x_max=x_max
    plot_data.y_min=y_min
    plot_data.y_max=y_max

    plot_data.update()
}
