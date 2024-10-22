
function laser_autofocus_debug_measure(){
    const num_images=document.getElementById("laser-autofocus-debug-num-images").value
    const total_z_um=document.getElementById("laser-autofocus-debug-total-z-um").value

    console.log("run autofocus debug with num image",num_images,"total z um",total_z_um)

    const trace1={
        x:[],
        y:[],
        mode:"lines+markers"
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
        mode:"lines+markers"
    }

    console.log("starting autofocus debug measurements")

    const clear_z_backlash_distance_mm=0.04
    immediate_move(1,"z",bottom_move_dist_um*1e-3,false)
    immediate_move(1,"z",-clear_z_backlash_distance_mm,false)
    immediate_move(1,"z",clear_z_backlash_distance_mm,false)
    for(let i=0;i<num_images;i++){
        if(i>0){
            immediate_move(1,"z",step_size_um*1e-3,false)
        }
        const current_offset=measureLaserAutofocusOffset(true)
        trace2.y[i]=current_offset
    }
    immediate_move(1,"z",bottom_move_dist_um*1e-3,false)
    console.log("done with autofocus debug measurements")

    const data=[trace1,trace2]
    const layout={
        title:"laser autofocus debug plot",
        autosize:true,

        margin: {
            t:30, // top margin for pan/zoom buttons
            l:30, // reduced y axis margin
            b:40, // bottom margin for x-axis title
        },
    }
    Plotly.newPlot("laser-autofocus-debug-plot",data,layout,{responsive: true})
}