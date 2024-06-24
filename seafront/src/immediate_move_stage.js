/**
 * 
 * @param {number} dist_sign 
 */
function move_x(dist_sign){
    let x_move_distance_mm_el=document.getElementById("x_move_distance_mm")
    if(!(x_move_distance_mm_el instanceof HTMLInputElement))throw new Error("element not found")
        
    let data={
        axis:"x",
        distance_mm:dist_sign*parseFloat(x_move_distance_mm_el.value)
    }

    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving x",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error moving x")
        })
        .send("/api/action/move_by",data,"POST")
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

    let data={
        axis:"y",
        distance_mm:dist_sign*parseFloat(y_move_distance_mm_el.value)
    }

    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving y",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error moving y")
        })
        .send("/api/action/move_by",data,"POST")
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

    let data={
        axis:"z",
        distance_mm:dist_sign*parseFloat(z_move_distance_um_el.value)*1e-3
    }

    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving z",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error moving z")
        })
        .send("/api/action/move_by",data,"POST")
}
/* callback on button pres to move forward in z */
function forward_z(){
    move_z(1)
}
/* callback on button pres to move backward in z */
function backward_z(){
    move_z(-1)
}