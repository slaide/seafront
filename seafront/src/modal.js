
/**
 * 
 * @param {string} title
 * @param {string|HTMLElement} html
 * @param {{oninit?:((modal:HTMLElement)=>void)}?} options
 * @returns {HTMLElement}
 */ 
function spawnModal(title,html,options=null){
    let modal=document.createElement("div")
    modal.classList.add("modal")

    let modal_content=document.createElement("div")
    modal_content.classList.add("modal-content")
    modal.appendChild(modal_content)

    let title_container=document.createElement("h2")
    title_container.innerHTML=title
    modal_content.appendChild(title_container)

    let close_button=document.createElement("button")
    close_button.innerText="Close"
    close_button.onclick=function(){
        document.body.removeChild(modal)
    }
    modal_content.appendChild(close_button)

    let html_body=document.createElement("div")
    if(html instanceof HTMLElement){
        html_body.appendChild(html)
    }else{
        html_body.innerHTML=html
    }
    modal_content.appendChild(html_body)

    if(options!=null){
        if(options.oninit!=null){
            options.oninit(modal_content)
        }
    }

    return document.body.appendChild(modal)
}