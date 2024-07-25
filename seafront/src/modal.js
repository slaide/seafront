/** @type{HTMLElement?} */
let modal_element=null
/** @type{{title:string,extra_buttons:{title:string,onclick:()=>void}[]}} */
let modal_data={
    title:"",
    extra_buttons:[]
}

/** @param body {string|HTMLElement} */
function modal_set_body(body){
    if(!modal_element) throw new Error("modal element not found")

    let modal_element_body=modal_element.children[0].children[1]
    if(!modal_element_body) throw new Error("modal body not found")

    // clear previous body
    while(modal_element_body.firstChild){modal_element_body.removeChild(modal_element_body.firstChild)}

    if(body instanceof HTMLElement){
        modal_element_body.appendChild(body)
    }else{
        modal_element_body.innerHTML=body
    }
}
/** @param {HTMLElement} element */
function set_reference_modal(element){
    modal_data=_p.manage(modal_data)

    // copy element reference to global scope
    modal_element=element
    if(!modal_element || !modal_element.parentElement) throw new Error("modal not found or no model parent exists")

    // remove modal element from DOM
    modal_element.removeAttribute("style")
    modal_element.parentElement.removeChild(modal_element)
}

/**
 * 
 * @param {string} title
 * @param {string|HTMLElement} body
 * @param {{oninit?:()=>void,buttons?:[{title:string,onclick:()=>void}]}?} options
 * @returns {HTMLElement}
 */ 
function spawnModal(title,body,options=null){
    if(!modal_element) throw new Error("modal element not found")

    modal_data.title=title

    //@ts-ignore
    modal_data.extra_buttons.length=0
    if(options!=null && options.buttons!=null){
        //@ts-ignore
        modal_data.extra_buttons.splice(0,0,...options.buttons)
    }

    //@ts-ignore
    modal_set_body(body)

    if(options!=null){
        if(options.oninit!=null){
            options.oninit()
        }
    }

    return document.body.appendChild(modal_element)
}