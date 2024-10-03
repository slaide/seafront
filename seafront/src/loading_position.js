
function enterLoadingPosition(){
    const data={}

    try{
        progress_indicator.run("Entering loading position")
    }catch(e){
        message_open("error","cannot currently enter loading position",e)
        return
    }
    
    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                message_open("error","error entering loading position",data)
            }

            progress_indicator.stop()
        })
        .onerror(function(){
            message_open("error","error entering loading position")

            progress_indicator.stop()
        })
        .send("/api/action/enter_loading_position",data,"POST")
}
function leaveLoadingPosition(){
    const data={}

    try{
        progress_indicator.run("leaving loading position")
    }catch(e){
        message_open("error","cannot currently leave loading position",e)
        return
    }

    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                message_open("error","error leaving loading position",data)
            }

            progress_indicator.stop()
        })
        .onerror(function(){
            message_open("error","error leaving loading position")

            progress_indicator.stop()
        })
        .send("/api/action/leave_loading_position",data,"POST")
}