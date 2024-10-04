let acquisition_progress=_p.manage({
    text:"no acquisition started",
    time_since_start_s:"",
    estimated_time_remaining_msg:"",
    acquisition_id:null
})

function start_acquisition(){
    let data={
        // acquisition config
        "config_file":_p.getUnmanaged(microscope_config)
    }
    
    let progress_element=document.getElementById("acquisition-progress-bar")
    if(!progress_element)throw new Error("progress_element is null")

    // make a raw copy of data to avoid modifying the original data
    data=JSON.parse(JSON.stringify(data))

    data.config_file.plate_wells=data.config_file.plate_wells
        // remove those that are not selected (also removes the headers)
        .filter(well=>well.selected)
        // adjust row and well to be 0-indexed (0 index in frontend is used to denote headers, on the server 0 is used to denote the first row/col)
        .map(well=>(new WellIndex(well.row-1,well.col-1,well.selected)))

    data.config_file.channels=data.config_file.channels
        .filter(channel=>channel.enabled)

    new XHR(false)
        .onload((xhr)=>{
            const data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                message_open("error","acquisition start failed because:",data.message)
                return
            }

            acquisition_progress.acquisition_id=data.acquisition_id
            message_open("info","acquisition started with id:",acquisition_progress.acquisition_id," ; response text: ",xhr.responseText)
            
            /** @type{XHR?} */
            let last_request=null

            const send_data={"acquisition_id":acquisition_progress.acquisition_id}

            if(!progress_element)throw new Error("progress_element is null")
            const updateIntervalHandle=setInterval(function(){
                last_request=new XHR()
                    .onload((xhr)=>{
                        if(last_request!=null && last_request.xhr!=xhr)
                            return;

                        if(!progress_element)throw new Error("progress_element is null")

                        let progress=JSON.parse(xhr.responseText)
                        if(progress.status!="success"){
                            acquisition_progress.text="error - "+progress.message
                            message_open("error","no acquisition progress available because ",progress.message)
                            clearInterval(updateIntervalHandle)
                            return
                        }
                        if(progress.acquisition_progress==null || progress.acquisition_meta_information==null){
                            acquisition_progress.text="running"
                            return
                        }

                        acquisition_progress.time_since_start_s=progress.acquisition_progress.time_since_start_s

                        // called estimated_total_time_s but actually contains the _remaining_ time in s
                        let remaining_time_s_total=parseFloat(progress.acquisition_progress.estimated_total_time_s)

                        let minutes=remaining_time_s_total%3600
                        let hours=(remaining_time_s_total-minutes)/3600
                        let seconds=minutes%60
                        minutes=(minutes-seconds)/60

                        let remaining_time_s=seconds.toFixed(0)
                        let remaining_time_m=minutes.toFixed(0)
                        let remaining_time_h=hours.toFixed(0)

                        /**
                         * @param{string} s
                         * @return{string}
                         */
                        function pad_to_two_digits(s){
                            if(s.length<2){
                                return "0"+s
                            }
                            return s
                        }

                        let time_remain_estimate_msg_string=""
                        if(remaining_time_s_total>0){
                            time_remain_estimate_msg_string="done in "
                            if(hours>0){
                                time_remain_estimate_msg_string+=remaining_time_h+"h:"
                            }
                            if(minutes>0){
                                if(hours>0){
                                    time_remain_estimate_msg_string+=pad_to_two_digits(remaining_time_m)+"m:"
                                }else{
                                    time_remain_estimate_msg_string+=remaining_time_m+"m:"
                                }
                            }
                            if(minutes>0){
                                time_remain_estimate_msg_string+=pad_to_two_digits(remaining_time_s)+"s"
                            }else{
                                time_remain_estimate_msg_string+=remaining_time_s+"s"
                            }
                        }
                        acquisition_progress.estimated_time_remaining_msg=time_remain_estimate_msg_string

                        acquisition_progress.text="running - "+progress.message
                        progress_element.style.setProperty(
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
                            acquisition_progress.text="done"
                        }
                    })
                    .onerror(()=>{
                        message_open("error","error getting acquisition progress")
                    })
                    .send("/api/acquisition/status",send_data,"POST")
            },1e3/5)
        })
        .onerror(()=>{
            message_open("error","error starting acquisition")
        })
        .send("/api/acquisition/start",data,"POST")
}
function cancel_acquisition(){
    const send_data={acquisition_id:acquisition_progress.acquisition_id}

    new XHR()
        .onload((xhr)=>{
            const data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                message_open("error","acquisition cancel failed because: ",data.message)
                return
            }
        })
        .onerror(()=>{
            message_open("error","error cancelling acquisition")
        })
        .send("/api/acquisition/cancel",send_data,"POST")
}