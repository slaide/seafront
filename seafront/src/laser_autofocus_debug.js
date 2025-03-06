
async function laser_autofocus_debug_measure(){
    const debug_num_images_element=document.getElementById("laser-autofocus-debug-num-images")
    const total_z_um_element=document.getElementById("laser-autofocus-debug-total-z-um")

    if(!(debug_num_images_element instanceof HTMLInputElement))throw Error()
    if(!(total_z_um_element instanceof HTMLInputElement))throw Error()

    const num_images=parseInt(debug_num_images_element.value)
    const total_z_um=parseFloat(total_z_um_element.value)
    if(num_images==null)throw Error()
    if(total_z_um==null)throw Error()

    console.log("run autofocus debug with num image",num_images,"total z um",total_z_um)

    /**@type {{x:number[],y:number[],mode:string,name:string}}*/
    const trace1={
        x:[],
        y:[],
        mode:"lines+markers",
        name:"real",
    }

    const bottom_move_dist_um=(-total_z_um/2)
    const step_size_um=total_z_um/(num_images-1)

    for(let i=0;i<num_images;i++){
        const target_z_value=bottom_move_dist_um+i*total_z_um/(num_images-1)

        trace1.x.push(target_z_value)
        trace1.y.push(target_z_value)
    }

    const trace2={
        x:trace1.x,
        y:trace1.y.map(v=>v),
        mode:"lines+markers",
        name:"measured",
    }

    const clear_z_backlash_distance_mm=40*1e-3
    await immediate_move(1,"z",bottom_move_dist_um*1e-3)
    await immediate_move(1,"z",-clear_z_backlash_distance_mm)
    await immediate_move(1,"z",clear_z_backlash_distance_mm)
    for(let i=0;i<num_images;i++){
        if(i>0){
            await immediate_move(1,"z",step_size_um*1e-3)
        }

        let current_offset=parseFloat("nan")
        try{
            const measured_offset=await measureLaserAutofocusOffset()

            if(measured_offset!=null && isFinite(measured_offset))current_offset=measured_offset
        }catch(e){}

        trace2.y[i]=current_offset
    }
    await immediate_move(1,"z",bottom_move_dist_um*1e-3)

    const trace3={
        x:trace1.x,
        y:trace1.y.map((val1,ind)=>{
            let val2=trace2.y[ind]
            return val1-val2
        }),
        mode:"lines+markers",
        name:"error"
    }

    const data=[trace1,trace2,trace3]
    const layout={
        title:"laser autofocus debug plot",
        autosize:true,

        yaxis: {
            title:"measured/estimated z",
            autorange: true,
        },
        xaxis:{
            title: "real z",
            autorange:true,
        },

        margin: {
            t:30, // top margin for pan/zoom buttons
            l:30, // reduced y axis margin
            b:40, // bottom margin for x-axis title
        },
    }
    Plotly.newPlot("laser-autofocus-debug-plot",data,layout,{responsive: true})
}