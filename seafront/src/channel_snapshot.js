/**
 * 
 * @param {ImagingChannel} channel 
 */
function snapChannel(channel){
    let data={
        machine_config:getConfigState(),
        channel:channel
    }
    
    new XHR()
        .onload((xhr)=>{
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error snapping channel",response)
                return
            }
        })
        .onerror(()=>{
            console.error("error snapping channel")
        })
        .send("/api/action/snap_channel",data,"POST")
}

/**
 * 
 * @param {HTMLInputElement} input_element 
 * @param {number} default_num_digits
 */
function ensureInputLimits(input_element,default_num_digits=2){
    let current_value=parseFloat(input_element.value)||0

    const min_value_attribute=input_element.getAttribute("min")
    if(min_value_attribute!=null){
        const min_value=parseFloat(min_value_attribute)
        if(current_value<min_value){
            current_value=min_value
        }
    }

    const max_value_attribute=input_element.getAttribute("max")
    if(max_value_attribute!=null){
        const max_value=parseFloat(max_value_attribute)
        if(current_value>max_value){
            current_value=max_value
        }
    }

    let num_digits=default_num_digits

    // derive num_digits from step attribute, if present
    const step_attribute=input_element.getAttribute("step")
    if(step_attribute!=null){
        const step=parseFloat(step_attribute)

        let remainder=current_value%step
        if(remainder!=0){
            let new_value=current_value-remainder
            if(remainder>step/2)
                new_value+=step
                current_value=new_value
        }

        // get num digits from step value, e.g. 0.x -> 1, 0.0x -> 2, 0.00x -> 3
        // or generally: xe-n -> n
        num_digits=Math.max(0,Math.ceil(-Math.log10(step)))
    }

    input_element.value=current_value.toFixed(num_digits)
}
