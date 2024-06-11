
function enterLoadingPosition(){
    new XHR(true)
        .onload(function(xhr){
            console.log("left loading position")
        })
        .onerror(function(){
            console.error("error entering loading position")
        })
        .send("/api/action/enter_loading_position",null,"POST")
}
function leaveLoadingPosition(){
    new XHR(true)
        .onload(function(xhr){
            console.log("left loading position")
        })
        .onerror(function(){
            console.error("error entering loading position")
        })
        .send("/api/action/leave_loading_position",null,"POST")
}