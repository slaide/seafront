
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
        .onload(async (xhr)=>{
            progress_indicator.stop()

            await fetch_image("laser_autofocus").then(imagedata=>{
                let img=document.getElementById("view_af_image")
                if(!(img instanceof HTMLCanvasElement))throw new Error("element with id 'view_af_image' is not an image element")

                img.height=imagedata.height
                img.width=imagedata.width

                img.getContext("2d")?.putImageData(imagedata,0,0)
            })
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            message_open("error","error snapping reflection autofocus")
        }).send("/api/action/snap_reflection_autofocus",data,"POST")
}