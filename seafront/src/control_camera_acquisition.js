
function start_streaming(){
    /** @type{object&{channel?:any}} */
    let data=getConfigState()

    let streaming_channel_select_element=document.getElementById("streaming-channel-select")
    if(!(streaming_channel_select_element instanceof HTMLSelectElement)){throw new Error("streaming-channel-select not found")}
    let streaming_channel_handle=streaming_channel_select_element.value
    for(let channel of microscope_config.channels){
        if(channel.handle==streaming_channel_handle){
            data.channel=channel
            break
        }
    }
    if(!data.channel){
        window.alert("channel "+streaming_channel_handle+" not found")
    }
    new XHR(true)
        .onload(function(xhr){
            console.log("success")
        })
        .onerror(function(){
            console.log("error streaming channel "+streaming_channel_handle)
        })
        .send("api/action/stream_channel_begin",data,"POST")
}
function stop_streaming(){
    /** @type{object&{channel?:any}} */
    let data=getConfigState()

    let streaming_channel_select_element=document.getElementById("streaming-channel-select")
    if(!(streaming_channel_select_element instanceof HTMLSelectElement)){throw new Error("streaming-channel-select not found")}
    let streaming_channel_handle=streaming_channel_select_element.value
    for(let channel of microscope_config.channels){
        if(channel.handle==streaming_channel_handle){
            data.channel=channel
            break
        }
    }
    if(!data.channel){
        window.alert("channel "+streaming_channel_handle+" not found")
    }
    new XHR(true)
        .onload(function(xhr){
            console.log("success")
        })
        .onerror(function(){
            console.log("error streaming channel "+streaming_channel_handle)
        })
        .send("api/action/stream_channel_end",data,"POST")
}