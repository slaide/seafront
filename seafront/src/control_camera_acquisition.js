
function start_streaming(){
    let data={
        machine_config:getConfigState(),
        framerate_hz:5,
        channel:null
    }

    let streaming_channel_select_element=document.getElementById("streaming-channel-select")
    if(!(streaming_channel_select_element instanceof HTMLSelectElement)){throw new Error("streaming-channel-select not found")}
    let streaming_channel_handle=streaming_channel_select_element.value
    for(let channel of microscope_config.channels){
        if(channel.handle==streaming_channel_handle){
            data.channel=channel
            break
        }
    }

    const framerate_element=document.getElementById("streaming-framerate-hz")
    if(!(framerate_element instanceof HTMLInputElement)){throw new Error("streaming-framerate-hz not found")}
    data.framerate_hz=parseFloat(framerate_element.value)

    if(!data.channel){
        window.alert("channel "+streaming_channel_handle+" not found")
    }

    microscope_state.streaming=true

    new XHR(true)
        .onload(function(xhr){
            const response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error starting stream "+xhr.responseText)
            }
        })
        .onerror(function(){
            console.error("error starting stream")
        })
        .send("api/action/stream_channel_begin",data,"POST")
}
function stop_streaming(){
    let data={
        machine_config:getConfigState(),
        channel:null
    }

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

    microscope_state.streaming=false
    
    new XHR(true)
        .onload(function(xhr){
            const response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("error stopping stream "+xhr.responseText)
            }
        })
        .onerror(function(){
            console.error("error stopping stream")
        })
        .send("api/action/stream_channel_end",data,"POST")
}