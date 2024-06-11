/**
 * 
 * @param {HardwareConfigItem} item 
 * @returns 
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

let machine_value_filter=_p.manage({value:""})
/**
 * 
 * @param {string} name
 * @returns 
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
function filter_results(){
    filtered_machine_defaults.splice(0,filtered_machine_defaults.length,...machine_defaults.filter((item)=>match_name(item.name)))
}