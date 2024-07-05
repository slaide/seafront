class PlotItemData{
    /**
     * container for plot item data (mainly axis limits)
     * @param {number} x_min 
     * @param {number} x_max 
     * @param {number} y_min 
     * @param {number} y_max 
     * @param {number} width
     * @param {number} height 
     */
    constructor(x_min,x_max,y_min,y_max,width,height){
        this.x_min=x_min
        this.x_max=x_max
        this.y_min=y_min
        this.y_max=y_max
        this.width=width
        this.height=height
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

        const width_attribute=element.getAttribute("width")
        if(width_attribute){
            width=parseFloat(width_attribute)
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

        const height_attribute=element.getAttribute("height")
        if(height_attribute){
            height=parseFloat(height_attribute)
        }

        return height
    }
}

class PlotData extends PlotItemData{
    /**
     * container for plot data (mainly axis limits)
     * @param {number} x_min 
     * @param {number} x_max 
     * @param {number} y_min 
     * @param {number} y_max 
     * @param {number} width
     * @param {number} height 
     * @param {HTMLElement[]} elements elements to share this plot data with
     */
    constructor(x_min,x_max,y_min,y_max,width,height,elements=[]){
        super(x_min,x_max,y_min,y_max,width,height)
        this.elements=elements
    }

    /**
     * construct plot data for a given element
     * @param {HTMLElement} element
     * @returns {PlotData}
     */
    static constructForElement(element){
        let initial_x_min=parseFloat(element.getAttribute("plot-x-min")||"nan")
        let initial_x_max=parseFloat(element.getAttribute("plot-x-max")||"nan")
        let initial_y_min=parseFloat(element.getAttribute("plot-y-min")||"nan")
        let initial_y_max=parseFloat(element.getAttribute("plot-y-max")||"nan")

        // if initial_x_min is nan, set to 0
        if(isNaN(initial_x_min)){
            initial_x_min=0
        }
        // if initial_x_max is nan, set to initial_x_min+child.clientWidth
        const element_width=this.getClientWidth(element)
        if(isNaN(initial_x_max)){
            initial_x_max=initial_x_min+element_width
        }
        // if initial_y_min is nan, set to 0
        if(isNaN(initial_y_min)){
            initial_y_min=0
        }
        // if initial_y_max is nan, set to initial_y_min+child.clientHeight
        const element_height=this.getClientHeight(element)
        if(isNaN(initial_y_max)){
            initial_y_max=initial_y_min+element_height
        }

        return new PlotData(initial_x_min,initial_x_max,initial_y_min,initial_y_max,element_width,element_height,[element])
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
            this._elementPlotData.set(element,PlotData.constructForElement(element))
        }

        const ret=this._elementPlotData.get(element)
        if(!ret){throw new Error("unreachable")}

        return ret
    }

    update(){
        requestAnimationFrame(()=>{
            for(let element of this.elements){
                Plot._update(element)
            }
        })
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

class Plot{
    static zoom_speed=0.02
    static invert_scroll=true
    static resizeObserver = new ResizeObserver(entries=>{
        for(let entry of entries){
            if(!(entry.target instanceof HTMLElement)){continue}
            Plot.resize(entry.target)
        }
    })

    /**
     * central element containing references to all container update functions to have a single interval updater
     * @type{function[]}
     * */
    static containerUpdateFuncs=[]
    static intervalUpdater=setInterval(function(){
        for(let f of Plot.containerUpdateFuncs){
            f()
        }
    },1e3/30) // 1e3/x -> check x times per second if the plot needs updating

    /**
     * callback to resize plot when element size changes
     * @param {HTMLElement} plot
     */
    static resize(plot){
        const plot_data=PlotData.getFor(plot)

        plot_data.width=PlotData.getClientWidth(plot)
        plot_data.height=PlotData.getClientHeight(plot)

        plot_data.update()
    }

    /**
     * update plot display given current state
     * @param {HTMLElement} plot 
     */
    static _update(plot){
        const plot_data=PlotData.getFor(plot)

        const scale_x=(plot_data.width/plot_data.getDeltaX())
        const scale_y=(plot_data.height/plot_data.getDeltaY())

        for(let child of plot.children){
            if(!(child instanceof HTMLElement)){continue}
            const child_plot_data=PlotData.getFor(child)

            // only change the attributes if they have changed, to avoid costly DOM updates

            const current_scale_x=child.style.getPropertyValue("--scale-x")
            const new_scale_x=scale_x+""
            if(current_scale_x!==new_scale_x){
                child.style.setProperty("--scale-x",new_scale_x)
            }
            const current_scale_y=child.style.getPropertyValue("--scale-y")
            const new_scale_y=scale_y+""
            if(current_scale_y!==new_scale_y){
                child.style.setProperty("--scale-y",new_scale_y)
            }

            const current_left=child.style.getPropertyValue("--left")
            const new_left=(child_plot_data.x_min-plot_data.x_min)*scale_x+"px"
            if(current_left!==new_left){
                child.style.setProperty("--left",new_left)
            }

            const current_bottom=child.style.getPropertyValue("--bottom")
            const new_bottom=(child_plot_data.y_min-plot_data.y_min)*scale_y+"px"
            if(current_bottom!==new_bottom){
                child.style.setProperty("--bottom",new_bottom)
            }
        }
    }

    /**
     * handle zoom (scroll/wheel) event on plot
     * @param {WheelEvent} event 
     */
    static plot_zoom(event){
        event.preventDefault()

        const plot = event.currentTarget
        if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

        let zoom_delta=event.deltaY
        const MAX_ZOOM_DELTA=5
        if(Math.abs(zoom_delta)>MAX_ZOOM_DELTA){
            zoom_delta=Math.sign(zoom_delta)*MAX_ZOOM_DELTA
        }

        // scrolling up naturally zooms out
        let zoom=Plot.zoom_speed*zoom_delta
        if(Plot.invert_scroll){
            zoom*=-1
        }
        
        let plot_data=PlotData.getFor(plot)
        const x_range = plot_data.getDeltaX()
        const y_range = plot_data.getDeltaY()

        // zoom in on cursor position with the frame
        const rect=plot.getBoundingClientRect()
        const x_frac=(event.clientX-rect.left)/rect.width
        const y_frac=(event.clientY-rect.top)/rect.height

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
    static plotDragInfo=new Map()
    /**
     * init plot dragging (handles mousedown)
     * @param {MouseEvent} event 
     */
    static plot_drag_start(event){
        // only start dragging on left click (0 is left click, according to mdn)
        if(event.button!=0)return
        // prevent default, which may e.g. browser initiated image drag
        event.preventDefault()

        if(!(event.currentTarget instanceof HTMLElement)){throw new Error("plot is not an html element")}
        Plot.plotDragInfo.set(event.currentTarget,{in_progress:true,x_start:event.clientX,y_start:event.clientY})
    }
    /**
     * process mouse move while dragging plot
     * @param {MouseEvent} event 
     */
    static plot_drag_move(event){
        let plot = event.currentTarget
        if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

        let plotDragData=Plot.plotDragInfo.get(plot)
        if(plotDragData && plotDragData.in_progress){
            event.preventDefault()

            const plot_data=PlotData.getFor(plot)
            const x_range = plot_data.getDeltaX()
            const y_range = plot_data.getDeltaY()

            const plot_x_size=PlotData.getClientWidth(plot)
            const plot_y_size=PlotData.getClientHeight(plot)

            const scale_x=(plot_x_size/x_range)
            const scale_y=(plot_y_size/y_range)

            const x_delta = event.clientX-plotDragData.x_start
            const y_delta = event.clientY-plotDragData.y_start

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
    static plot_drag_end(event){
        let plot = event.currentTarget
        if(!(plot instanceof HTMLElement)){throw new Error("plot is not an html element")}

        let plotDragData=Plot.plotDragInfo.get(plot)
        if(!plotDragData){return}

        plotDragData.in_progress=false
        Plot.plotDragInfo.delete(plot)
    }

    /**
     * init element as plot display
     * @param {HTMLElement} plot 
     * @param {(object&{preserveAspectRatio:boolean})?} options
     */
    static plot_init(plot,options=null){
        plot.addEventListener("wheel",Plot.plot_zoom)
        plot.addEventListener("mousedown",Plot.plot_drag_start)
        plot.addEventListener("mousemove",Plot.plot_drag_move)
        plot.addEventListener("mouseup",Plot.plot_drag_end)
        plot.addEventListener("mouseleave",Plot.plot_drag_end)
        plot.addEventListener("dblclick",Plot.plot_fit)
        
        Plot.resizeObserver.observe(plot)

        let plotHasChanged=false
        Plot.containerUpdateFuncs.push(function(){
            if(plotHasChanged){
                plotHasChanged=false
                const plot_data=PlotData.getFor(plot)

                plot_data.update()
            }
        })

        // set callback for attribute change on child to update corresponding data and update the plot
        const attribute_change_observer = new MutationObserver((mutationsList, observer) => {
            for(let mutation of mutationsList){
                if(
                    mutation.type==="attributes"
                    && mutation.attributeName!=null
                    && (
                        ["width","height","plot-x-min","plot-x-max","plot-y-min","plot-x-max"]
                        .indexOf(mutation.attributeName) > -1
                    )
                ){
                    const child=mutation.target
                    if(!(child instanceof HTMLElement))throw new Error("unreachable")

                    const child_data=PlotData.getFor(child)
                    child.style.setProperty("--width",child_data.width+"px");
                    child.style.setProperty("--height",child_data.height+"px");

                    PlotData._elementPlotData.set(child,PlotData.constructForElement(child))

                    // request plot update
                    plotHasChanged=true
                }
            }
        })

        /** @param {HTMLElement} child */
        function processChild(child){
            // ensure there is plot data for each child
            const child_plot_data=PlotData.getFor(child)

            const child_width=child_plot_data.width
            const child_height=child_plot_data.height

            child.style.setProperty("--width",child_width+"px");
            child.style.setProperty("--height",child_height+"px");
            child.classList.add("plot-img")
            
            attribute_change_observer.observe(child, {attributes:true})
        }

        for(let child of plot.children){
            if(!(child instanceof HTMLElement)){continue}
            processChild(child)
        }

        let preserveAspectRatio=options?.preserveAspectRatio||false
        if(preserveAspectRatio){
            // TODO store this on an element
        }

        // register callback for new children added after this function has run
        const add_child_observer = new MutationObserver((mutationsList, observer) => {
            for(let mutation of mutationsList){
                if(mutation.type==="childList"){
                    for(let node of mutation.addedNodes){
                        if(node instanceof HTMLElement){
                            processChild(node)
                            
                            // request plot update
                            plotHasChanged=true
                        }
                    }
                }
            }
        })
        add_child_observer.observe(plot, {childList:true})

        Plot.plot_fit(plot)
    }
    /**
     * fit plot to display all children
     * @param {Event|HTMLElement} plot 
     */
    static plot_fit(plot){
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

        // adjust limits to ensure aspect ratio consistent with plot_element size
        let plot_width=plot_data.width
        let plot_height=plot_data.height

        // aspect ratio is width/height -> larger aspect ratio -> more x per y
        let plot_native_aspect_ratio=plot_width/plot_height
        let current_axis_aspect_ratio=(x_max-x_min)/(y_max-y_min)

        if(plot_native_aspect_ratio>current_axis_aspect_ratio){
            // if plot element is wider than the current x axis, increase x axis
            let x_center=(x_max+x_min)/2
            let x_range=(y_max-y_min)*plot_native_aspect_ratio
            x_min=x_center-x_range/2
            x_max=x_center+x_range/2
        }else{
            // if plot is taller than the current y axis, increase y axis
            let y_center=(y_max+y_min)/2
            let y_range=(x_max-x_min)/plot_native_aspect_ratio
            y_min=y_center-y_range/2
            y_max=y_center+y_range/2
        }

        plot_data.x_min=x_min
        plot_data.x_max=x_max
        plot_data.y_min=y_min
        plot_data.y_max=y_max

        plot_data.update()
    }
}

/**
 * this is used as a p:init callback
 * @param {HTMLElement} element 
 */
function linkPlots(element){
    let plots=element.querySelectorAll(".channel-plot-display")
    if(plots.length==0)throw new Error("no channel-plot-display elements found")

    // @ts-ignore
    let plot_data=PlotData.getFor(plots[0])
    for(let plot_element of plots){
        // @ts-ignore
        plot_data.link(plot_element)
    }
}