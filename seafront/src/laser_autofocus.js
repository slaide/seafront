
let laserautofocusdata=_p.manage({currentOffset:0.0})

/**
 * @param{HTMLElement} element
 */
function laser_autofocus_initcheckbox(element){
    _p.onValueChangeCallback(()=>{
        const machine_config_calibration_item=getMachineConfig("laser_autofocus_is_calibrated")
        if(machine_config_calibration_item==null){throw new Error("laser_autofocus_is_calibrated not found in machine config")}
        return machine_config_calibration_item.value=="yes"
    },(calibration_present)=>{
        if(calibration_present){
            element.removeAttribute("disabled")
        }else{
            element.setAttribute("disabled","")
            microscope_config.autofocus_enabled=false
        }
    })
}

/**
 * measure autofocus offset
 * 
 * if return_immediate is true, this call is sync and returns the offset in um
 * @returns {number?}
 */
function measureLaserAutofocusOffset(return_immediate=false){
    let data={
        config_file:_p.getUnmanaged(microscope_config),
    }

    try{
        progress_indicator.run("Measuring laser autofocus offset")
    }catch(e){
        message_open("error","cannot currently measure autofocus offset",e)
        return null
    }

    /**@type{number?}*/
    let immediate_return_value=null

    new XHR(!return_immediate)
        .onload((xhr)=>{
            progress_indicator.stop()

            let response=JSON.parse(xhr.responseText)

            laserautofocusdata.currentOffset=response.displacement_um

            if(return_immediate){
                immediate_return_value=response.displacement_um
            }
        })
        .onerror((xhr)=>{
            progress_indicator.stop()
            
            message_open("error","error measuring laser autofocus offset",xhr.responseText)
        })
        .send("/api/action/measure_displacement",data,"POST")

    return immediate_return_value
}

function setLaserAutofocusReference(){
    let data={}

    try{
        progress_indicator.run("calibrating laser autofocus")
    }catch(e){
        message_open("error","cannot currently calibrate autofocus",e)
        return
    }
    
    new XHR(true)
        .onload((xhr)=>{
            progress_indicator.stop()
            
            /**@type{{status:string,calibration_data?:{x_reference:number,um_per_px:number,calibration_position:{x_pos_mm:number,y_pos_mm:number,z_pos_mm:number}}}}*/
            let response=JSON.parse(xhr.responseText)

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
        .onerror((xhr)=>{
            progress_indicator.stop()
            
            message_open("error","error setting laser autofocus reference",xhr.responseText)
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

    try{
        progress_indicator.run("Moving to target offset")
    }catch(e){
        message_open("error","cannot currently move to target offset",e)
        return
    }

    new XHR(true)
        .onload((xhr)=>{
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
        })
        .onerror((xhr)=>{
            progress_indicator.stop()
            
            message_open("error","error moving to target offset",xhr.responseText)
        })
        .send("/api/action/laser_autofocus_move_to_target_offset",data,"POST")
}