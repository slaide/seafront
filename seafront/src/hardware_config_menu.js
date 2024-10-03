/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns {"number"|"text"}
 */
function getInputType(item){
    if(item.value_kind=="number"){
        return "number"
    }else if(item.value_kind=="text"){
        return "text"
    }else{
        throw new Error("unknown type "+item.value_kind)
    }
}
/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns 
 */
function itemIsRawInput(item){
    return item.value_kind=="number" || item.value_kind=="text"
}
/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns 
 */
function itemIsOptionInput(item){
    return item.value_kind=="option"
}
/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns 
 */
function itemIsAction(item){
    return item.value_kind=="action"
}
/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns 
 */
function executeActionItem(item){
    if(!itemIsAction(item)){throw new Error("item is not an action")}

    const data={}

    try{
        progress_indicator.run("machine action "+item.name)
    }catch(e){
        message_open("error","cannot currently execute action",e)
        return
    }

    // send xhr with item.value as url
    new XHR(true)
        .onload((xhr)=>{
            progress_indicator.stop()
            
            let response=JSON.parse(xhr.responseText)
            if(response.status!="success"){
                console.error("action failed",item,response)
            }
        })
        .onerror(()=>{
            progress_indicator.stop()
            
            console.error("action failed",item)
        })
        // @ts-ignore
        .send(item.value,data,"POST")
}

let machine_value_filter=_p.manage({value:""})
/**
 * check if name contains the filter query
 * @param {string} name
 * @returns {boolean}
 */
function match_name(name){
    if(machine_value_filter.value.length==0)
        return true;

    const lowercasename=name.toLowerCase()
    const lowercasemachine_value_filter=machine_value_filter.value.toLowerCase()
    const ret=lowercasename.includes(lowercasemachine_value_filter)
    return ret
}
/** @type{HardwareConfigItem[]} */
let filtered_machine_defaults=_p.manage([])

/**
 * apply the filter to the machine_defaults and store the result in filtered_machine_defaults
 * 
 * used as callback (on the search bar)
 */
function filter_results(){
    filtered_machine_defaults.length=0
    filtered_machine_defaults.splice(0,0,...microscope_config.machine_config.filter((item)=>match_name(item.name)))
}