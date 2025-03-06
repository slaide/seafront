let message_last_id=0;

/** @typedef {{text:string,id:number,level:"info"|"warn"|"error",level_string:string,timestamp:string}} Message */

/** @type {Message[]} */
let messages=_p.manage([])
/**
 * @param {"warn"|"info"|"error"} level
 * @param {any} msg
 */
function message_open(level,...msg){
    const current_time=new Date()

    const timestring=current_time.toLocaleString(undefined,{
          hour:"numeric",
          minute:"numeric",
          second:"numeric"
        })
    const datestring=current_time.toLocaleString(undefined,{
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        })
    const timestamp_string=timestring+" on "+datestring

    let level_string=""
    if(level=="info"){
        console.log("info at "+timestamp_string+" - ",...msg)
        level_string="info"
    }
    if(level=="warn"){
        console.warn("warning at "+timestamp_string+" - ",...msg)
        level_string="warning"
    }
    if(level=="error"){
        console.error("error at "+timestamp_string+" - ",...msg)
        level_string="error"
    }

    let message=""
    for(let msg_part of msg){
        if(msg_part instanceof Error){
            message+="Error( "+msg_part.message+" )"
        }else if(typeof msg_part === "object" && msg_part != null){
            try{
                message+=JSON.stringify(msg_part)
            }catch(e){
                message+=msg_part
            }
        }else{
            message+=msg_part
        }
        message+=" "
    }
    messages.push({
        text:message,
        id:message_last_id+1,
        level:level,
        level_string:level_string,
        timestamp:timestamp_string
    })
    message_last_id+=1
}
/**
 * @param {number} target_message_id
 */
function message_close(target_message_id){
    let message_index=-1;

    let current_message_index=-1;
    for(let message of messages){
        current_message_index+=1;
        if(message.id==target_message_id){
            message_index=current_message_index;
        }
    }

    if(message_index<0)
        return;

    messages.splice(message_index,1)
}

/**
 * @param {Message} message
 */
function message_get_class(message){
    if(message.level=="info"){
        return "message_level_info"
    }
    if(message.level=="warn"){
        return "message_level_warn"
    }
    if(message.level=="error"){
        return "message_level_error"
    }
    throw Error("unknown level "+message.level)
}
/**
 * @param {Element} element
 * @param {Message} message
 */
function message_init(element,message){
    element.classList.add(message_get_class(message))
}