
function snapReflectionAutofocus(){
    new XHR(false)
        .onload((xhr)=>{
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
            console.error("error snapping reflection autofocus")
        })
        .send("/api/action/snap_reflection_autofocus",{"exposure_time_ms":100,"analog_gain":0},"POST")
}