
let laserautofocusdata=_p.manage({currentOffset:0.0})

function measureLaserAutofocusOffset(){
    let data={}

    progress_indicator.run("Measuring laser autofocus offset")

    new XHR()
        .onload((xhr)=>{
            progress_indicator.stop()

            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error measuring laser autofocus offset",response)
                return
            }
            laserautofocusdata.currentOffset=response.displacement_um
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("error measuring laser autofocus offset")
        })
        .send("/api/action/measure_displacement",data,"POST")
}

function setLaserAutofocusReference(){
    let data={}

    progress_indicator.run("calibrating laser autofocus")
    
    new XHR()
        .onload((xhr)=>{
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error setting laser autofocus reference",response)
                return
            }
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("error setting laser autofocus reference")
        })
        .send("/api/action/laser_af_calibrate",data,"POST")
}

function laserAutofocusMoveToTargetOffset(){
    const offset_element=document.getElementById("laser-autofocus-target-offset-um")
    if(!(offset_element instanceof HTMLInputElement)){console.error("laser autofocus target offset element not found");return}
    const target_offset_um=parseFloat(offset_element.value)
    const data={
        target_offset_um:target_offset_um
    }

    progress_indicator.run("Moving to target offset")

    new XHR(false)
        .onload((xhr)=>{
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error moving to target offset",response)
                return
            }
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("error moving to target offset")
        })
        .send("/api/action/laser_autofocus_move_to_target_offset",data,"POST")
}