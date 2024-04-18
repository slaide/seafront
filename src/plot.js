
/**
 * 
 * @param {HTMLElement} plot 
 */
p.plot_update=function(plot){
    let x_range = plot.x_max - plot.x_min;
    let y_range = plot.y_max - plot.y_min;

    let plot_x_size=plot.clientWidth;
    let plot_y_size=plot.clientHeight;

    let scale_x=(plot_x_size/x_range);
    let scale_y=(plot_y_size/y_range);

    for(let child of plot.children){
        let c_width=child.x_max-child.x_min;
        let c_height=child.y_max-child.y_min;

        child.style.setProperty("--scale-x",scale_x);
        child.style.setProperty("--scale-y",scale_y);

        child.style.setProperty("--left",(child.x_min-plot.x_min)*scale_x+"px");
        child.style.setProperty("--bottom",(child.y_min-plot.y_min)*scale_y+"px");
    }
}
let zoom_speed=0.01;
let invert_scroll=true;

/**
 * 
 * @param {WheelEvent} event 
 */
p.plot_zoom=function(event){
    event.preventDefault();

    let plot = event.currentTarget;

    let zoom_delta=event.deltaY;
    if(Math.abs(zoom_delta)>5){
        zoom_delta=Math.sign(zoom_delta)*5;
    }

    // scrolling up naturally zooms out
    let zoom=zoom_speed*zoom_delta;
    if(invert_scroll){
        zoom*=-1;
    }
    
    let x_range = plot.x_max - plot.x_min;
    let y_range = plot.y_max - plot.y_min;

    plot.x_min += x_range*zoom;
    plot.x_max -= x_range*zoom;
    plot.y_min += y_range*zoom;
    plot.y_max -= y_range*zoom;

    this.plot_update(plot);
}
/**
 * 
 * @param {MouseEvent} event 
 */
p.plot_drag_start=function(event){
    this.drag_in_progress=true;
    this.drag_x_start=event.clientX;
    this.drag_y_start=event.clientY;
}
/**
 * 
 * @param {MouseEvent} event 
 */
p.plot_drag_move=function(event){
    if(this.drag_in_progress){
        event.preventDefault();
        let plot = event.currentTarget;

        let x_range = plot.x_max - plot.x_min;
        let y_range = plot.y_max - plot.y_min;

        let plot_x_size=plot.clientWidth;
        let plot_y_size=plot.clientHeight;

        let scale_x=(plot_x_size/x_range);
        let scale_y=(plot_y_size/y_range);

        let x_delta = event.clientX-this.drag_x_start;
        let y_delta = event.clientY-this.drag_y_start;

        plot.x_min -= x_delta/scale_x;
        plot.x_max -= x_delta/scale_x;
        plot.y_min += y_delta/scale_y;
        plot.y_max += y_delta/scale_y;

        this.drag_x_start=event.clientX;
        this.drag_y_start=event.clientY;

        this.plot_update(plot);
    }
}

/**
 * 
 * @param {MouseEvent} event 
 */
p.plot_drag_end=function(event){
    this.drag_in_progress=false;
}
/**
 * 
 * @param {HTMLElement} plot 
 */
p.plot_init=function(plot){
    for(let child of plot.children){
        child.x_min=0;
        child.x_max=child.clientWidth;
        child.y_min=0;
        child.y_max=child.clientHeight;

        child.style.setProperty("--width",child.clientWidth+"px");
        child.style.setProperty("--height",child.clientHeight+"px");
        child.classList.add("plot-img")
    }

    this.plot_fit(plot)
}
/**
 * 
 * @param {Event} plot 
 */
p.plot_fit=function(plot){
    let plot_element=plot.currentTarget || plot
    // adjust plot to fit the children

    let x_min=Infinity;
    let x_max=-Infinity;
    let y_min=Infinity;
    let y_max=-Infinity;

    for(let child of plot_element.children){
        x_min=Math.min(x_min,child.x_min);
        x_max=Math.max(x_max,child.x_max);
        y_min=Math.min(y_min,child.y_min);
        y_max=Math.max(y_max,child.y_max);
    }

    plot_element.x_min=x_min;
    plot_element.x_max=x_max;
    plot_element.y_min=y_min;
    plot_element.y_max=y_max;

    this.plot_update(plot_element);
}
