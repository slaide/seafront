
function snapReflectionAutofocus(){
    const exposure_time_input_element=document.getElementById("laser-af-debug-exposure-time-ms")
    if(!(exposure_time_input_element instanceof HTMLInputElement))throw new Error("element with id 'laser-af-debug-exposure-time-ms' is not an input element")
    const exposure_time_ms=parseFloat(exposure_time_input_element.value)

    const analog_gain_input_element=document.getElementById("laser-af-debug-analog-gain")
    if(!(analog_gain_input_element instanceof HTMLInputElement))throw new Error("element with id 'laser-af-debug-analog-gain' is not an input element")
    const analog_gain=parseFloat(analog_gain_input_element.value)

    try{
        progress_indicator.run("Snapping reflection autofocus")
    }catch(e){
        message_open("error","snap currently reflection autofocus",e)
        return
    }

    let data={
        "exposure_time_ms":exposure_time_ms,
        "analog_gain":analog_gain,
    }

    new XHR(true)
        .onload((xhr)=>{
            progress_indicator.stop()

            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error snapping reflection autofocus",response)
                return
            }
            let handle=response.img_handle
            let img=document.getElementById("view_af_image")
            if(!(img instanceof HTMLImageElement))throw new Error("element with id 'view_af_image' is not an image element")

            if(response.width_px!=null)
                img.setAttribute("width",response.width_px)
            if(response.height_px!=null)
                img.setAttribute("height",response.height_px)
            img.src="/img/get_by_handle?img_handle="+handle
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("error snapping reflection autofocus")
        })
        .send("/api/action/snap_reflection_autofocus",data,"POST")
}