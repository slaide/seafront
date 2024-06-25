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

    // make a raw copy of data to avoid modifying the original data
    data=JSON.parse(JSON.stringify(data))

    data.config_file.plate_wells=data.config_file.plate_wells
        // remove those that are not selected (also removes the headers)
        .filter(well=>well.selected)
        // adjust row and well to be 0-indexed (0 index in frontend is used to denote headers, on the server 0 is used to denote the first row/col)
        .map(well=>(new WellIndex(well.row,well.col,well.selected)))

    data.config_file.channels=data.config_file.channels
        .filter(channel=>channel.enabled)

    new XHR(false)
        .onload((xhr)=>{
            const data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                console.error("acquisition start failed because:",data.message)
                return
            }

            let acquisition_id=data.acquisition_id
            console.log("acquisition started with id",acquisition_id,"response text:",xhr.responseText)
            
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

                                if(!progress_element)throw new Error("progress_element is null")
                                    if(progress_element.children.length<2)throw new Error("progress_element has no children")
                                    if(!(progress_element.children[1] instanceof HTMLElement))throw new Error("progress_element.children[1] is not an HTMLElement")

                                let progress=JSON.parse(xhr.responseText)
                                if(progress.status!="success"){
                                    progress_element.children[1].innerText="Acquisition status unknown"
                                    console.warn("no acquisition progress available because",progress.message)
                                    return
                                }

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
                                    progress_element.children[1].innerText="Acquisition is done"
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
