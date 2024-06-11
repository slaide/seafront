
let laserautofocusdata=_p.manage({currentOffset:0.0})

function measureLaserAutofocusOffset(){
    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error measuring laser autofocus offset",response)
                return
            }
            laserautofocusdata.currentOffset=response.displacement_um
        })
        .onerror(()=>{
            console.error("error measuring laser autofocus offset")
        })
        .send("/api/action/measure_displacement",null,"POST")
}

function setLaserAutofocusReference(){
    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error setting laser autofocus reference",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error setting laser autofocus reference")
        })
        .send("/api/action/laser_af_calibrate",null,"POST")
}

function laserAutofocusMoveToTargetOffset(){
    const offset_element=document.getElementById("laser-autofocus-target-offset-um")
    if(!(offset_element instanceof HTMLInputElement)){console.error("laser autofocus target offset element not found");return}
    const target_offset_um=parseFloat(offset_element.value)
    const data={
        target_offset_um:target_offset_um
    }

    new XHR(false)
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving to target offset",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error moving to target offset")
        })
        .send("/api/action/laser_autofocus_move_to_target_offset",data,"POST")
}