
/**
 * 
 * @param {number} sign 
 * @param {"x"|"y"|"z"} axis 
 * @param {number} distance_mm 
 * @returns {Promise<{}>|null}
 */
function immediate_move(sign,axis,distance_mm){
    let data={
        axis:axis,
        distance_mm:sign*distance_mm
    }

    try{
        progress_indicator.run("moving "+axis)
    }catch(e){
        message_open("error","cannot currently move",e)
        return null
    }

    return new Promise((resolve,reject)=>{
        new XHR(true)
            .onload((xhr)=>{
                progress_indicator.stop()
                
                let response=JSON.parse(xhr.responseText)
                resolve(response)
            })
            .onerror((xhr)=>{
                progress_indicator.stop()
                
                message_open("error","error moving "+axis,xhr.responseText)

                reject()
            })
            .send("/api/action/move_by",data,"POST")
    })
}

/**
 * 
 * @param {number} dist_sign 
 */
async function move_x(dist_sign){
    let x_move_distance_mm_el=document.getElementById("x_move_distance_mm")
    if(!(x_move_distance_mm_el instanceof HTMLInputElement))throw new Error("element not found")

    await immediate_move(dist_sign,"x",parseFloat(x_move_distance_mm_el.value))
}
/* callback on button pres to move forward in x */
async function forward_x(){
    await move_x(1)
}
/* callback on button pres to move backward in x */
async function backward_x(){
    await move_x(-1)
}

/**
 * 
 * @param {number} dist_sign 
 */
async function move_y(dist_sign){
    let y_move_distance_mm_el=document.getElementById("y_move_distance_mm")
    if(!(y_move_distance_mm_el instanceof HTMLInputElement))throw new Error("element not found")

    await immediate_move(dist_sign,"y",parseFloat(y_move_distance_mm_el.value))
}
/* callback on button pres to move forward in y */
async function forward_y(){
    await move_y(1)
}
/* callback on button pres to move backward in y */
async function backward_y(){
    await move_y(-1)
}

/**
 * 
 * @param {number} dist_sign 
 */
async function move_z(dist_sign){
    let z_move_distance_um_el=document.getElementById("z_move_distance_um")
    if(!(z_move_distance_um_el instanceof HTMLInputElement))throw new Error("element not found")

    await immediate_move(dist_sign,"z",parseFloat(z_move_distance_um_el.value)*1e-3)
}
/* callback on button pres to move forward in z */
async function forward_z(){
    await move_z(1)
}
/* callback on button pres to move backward in z */
async function backward_z(){
    await move_z(-1)
}