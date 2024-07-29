
let laserautofocusdata=_p.manage({currentOffset:0.0})

function measureLaserAutofocusOffset(){
    let data={
        config_file:_p.getUnmanaged(microscope_config),
    }

    progress_indicator.run("Measuring laser autofocus offset")

    new XHR(true)
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
    
    new XHR(true)
        .onload((xhr)=>{
            progress_indicator.stop()
            
            /**@type{{status:string,calibration_data?:{x_reference:number,um_per_px:number,calibration_position:{x_pos_mm:number,y_pos_mm:number,z_pos_mm:number}}}}*/
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error setting laser autofocus reference",response)
                return
            }

            const autofocus_enabled_checkbox_element=document.getElementById("autofocus-enabled-checkbox")
            if(!(autofocus_enabled_checkbox_element instanceof HTMLInputElement)){console.error("element not found");return}

            //@ts-ignore
            getMachineConfig("laser_autofocus_is_calibrated").value="yes"
            //@ts-ignore
            getMachineConfig("laser_autofocus_calibration_x").value=response.calibration_data.x_reference
            //@ts-ignore
            getMachineConfig("laser_autofocus_calibration_umpx").value=response.calibration_data.um_per_px

            //@ts-ignore
            getMachineConfig("laser_autofocus_calibration_refzmm").value=response.calibration_data.calibration_position.z_pos_mm

            // set ability to enable/disable autofocus
            autofocus_enabled_checkbox_element.disabled=false

            // enable use of autofocus upon calibration
            microscope_config.autofocus_enabled=true
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
        target_offset_um:target_offset_um,
        config_file:_p.getUnmanaged(microscope_config),
    }

    progress_indicator.run("Moving to target offset")

    new XHR(true)
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