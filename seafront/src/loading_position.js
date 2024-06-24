
function enterLoadingPosition(){
    const data={}
    
    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                console.error("error entering loading position",data)
            }
        })
        .onerror(function(){
            console.error("error entering loading position")
        })
        .send("/api/action/enter_loading_position",data,"POST")
}
function leaveLoadingPosition(){
    const data={}

    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                console.error("error leaving loading position",data)
            }
        })
        .onerror(function(){
            console.error("error leaving loading position")
        })
        .send("/api/action/leave_loading_position",data,"POST")
}