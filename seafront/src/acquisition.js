/**@type{HTMLElement?} */
let progress_element=null

/**
 * 
 * @param {HTMLElement} element 
 */
function removecontainerelement(element){
    progress_element=element
    element.parentElement?.removeChild(element)
}

function start_acquisition(){
    let data={
        // machine config
        "machine_config":getConfigState(),
        // acquisition config
        "config_file":_p.getUnmanaged(microscope_config)
    }
    console.log("acquisition with data:",data)

    new XHR(false)
        .onload((xhr)=>{
            let acquisition_id=JSON.parse(xhr.responseText).acquisition_id
            
            /** @type{XHR?} */
            let last_request=null

            if(!progress_element)throw new Error("progress_element is null")
            spawnModal("Acquisition Progress",progress_element,{
                oninit:function(){
                    const updateIntervalHandle=setInterval(function(){
                        last_request=new XHR()
                            .onload((xhr)=>{
                                if(last_request!=null && last_request.xhr!=xhr)
                                    return;

                                let progress=JSON.parse(xhr.responseText)

                                if(!progress_element)throw new Error("progress_element is null")
                                if(progress_element.children.length<2)throw new Error("progress_element has no children")
                                if(!(progress_element.children[1] instanceof HTMLElement))throw new Error("progress_element.children[1] is not an HTMLElement")
                                    
                                progress_element.children[1].innerText=progress.message
                                progress_element.children[1].style.setProperty(
                                    "--percent-done",
                                    (
                                        progress.acquisition_progress.current_num_images
                                        / progress.acquisition_meta_information.total_num_images
                                        * 100
                                    ) + "%"
                                )
                                
                                // stop polling when acquisition is done
                                if(progress.acquisition_progress.current_num_images==progress.acquisition_meta_information.total_num_images){
                                    clearInterval(updateIntervalHandle)
                                }
                            })
                            .onerror(()=>{
                                console.error("error getting acquisition progress")
                            })
                            .send("/api/acquisition/status",{"acquisition_id":acquisition_id},"POST")
                    },1e3/15)
                }
            })
        })
        .onerror(()=>{
            console.error("error starting acquisition")
        })
        .send("/api/acquisition/start",data,"POST")
}
