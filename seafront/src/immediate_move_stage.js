
/**
 * 
 * @param {number} sign 
 * @param {"x"|"y"|"z"} axis 
 * @param {number} distance_mm 
 */
function immediate_move(sign,axis,distance_mm){
    let data={
        axis:axis,
        distance_mm:sign*distance_mm
    }

    progress_indicator.run("moving "+axis)

    new XHR()
        .onload((xhr)=>{
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving "+axis,response)
                return
            }
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("error moving "+axis)
        })
        .send("/api/action/move_by",data,"POST")
}

/**
 * 
 * @param {number} dist_sign 
 */
function move_x(dist_sign){
    let x_move_distance_mm_el=document.getElementById("x_move_distance_mm")
    if(!(x_move_distance_mm_el instanceof HTMLInputElement))throw new Error("element not found")

    immediate_move(dist_sign,"x",parseFloat(x_move_distance_mm_el.value))
}
/* callback on button pres to move forward in x */
function forward_x(){
    move_x(1)
}
/* callback on button pres to move backward in x */
function backward_x(){
    move_x(-1)
}

/**
 * 
 * @param {number} dist_sign 
 */
function move_y(dist_sign){
    let y_move_distance_mm_el=document.getElementById("y_move_distance_mm")
    if(!(y_move_distance_mm_el instanceof HTMLInputElement))throw new Error("element not found")

    immediate_move(dist_sign,"y",parseFloat(y_move_distance_mm_el.value))
}
/* callback on button pres to move forward in y */
function forward_y(){
    move_y(1)
}
/* callback on button pres to move backward in y */
function backward_y(){
    move_y(-1)
}

/**
 * 
 * @param {number} dist_sign 
 */
function move_z(dist_sign){
    let z_move_distance_um_el=document.getElementById("z_move_distance_um")
    if(!(z_move_distance_um_el instanceof HTMLInputElement))throw new Error("element not found")

    immediate_move(dist_sign,"z",parseFloat(z_move_distance_um_el.value)*1e-3)
}
/* callback on button pres to move forward in z */
function forward_z(){
    move_z(1)
}
/* callback on button pres to move backward in z */
function backward_z(){
    move_z(-1)
}