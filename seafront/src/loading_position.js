
function enterLoadingPosition(){
    const data={}

    progress_indicator.run("Entering loading position")
    
    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                console.error("error entering loading position",data)
            }

            progress_indicator.stop()
        })
        .onerror(function(){
            console.error("error entering loading position")

            progress_indicator.stop()
        })
        .send("/api/action/enter_loading_position",data,"POST")
}
function leaveLoadingPosition(){
    const data={}

    progress_indicator.run("leaving loading position")

    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                console.error("error leaving loading position",data)
            }

            progress_indicator.stop()
        })
        .onerror(function(){
            console.error("error leaving loading position")

            progress_indicator.stop()
        })
        .send("/api/action/leave_loading_position",data,"POST")
}