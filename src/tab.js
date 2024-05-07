
/**
 * 
 * @param {HTMLElement} tab_header 
 * @returns 
 */
const init_tab_header=function(tab_header){
    let tab_header_children=tab_header.querySelectorAll("*[target]")
    
    /** @type HTMLElement[] */
    let valid_tab_children=[]
    tab_header_children.forEach((el)=>{
        if(!(el instanceof HTMLElement)){return}
        
        let element_target_id=el.getAttribute("target")
        if(!element_target_id){console.error("element target is null");return}
        let tab_target=document.getElementById(element_target_id)
        if(!tab_target){
            console.error("tab header target '"+el.getAttribute("target")+"' not found",el);
            return
        }
        tab_target.classList.add("hidden");

        valid_tab_children.push(el);
        el.addEventListener("click",tab_head_click);
    });
    if(valid_tab_children.length==0){
        return
    }
    valid_tab_children[0].click()
}
let _tabHeadMap_currentTarget=new Map()
/**
 * 
 * @param {MouseEvent} e 
 */
const tab_head_click=function(e){
    let head=e.currentTarget;
    if(!head){return}
    if(!(head instanceof HTMLElement)){return}
    if(!head.parentNode){return}

    let current_target=_tabHeadMap_currentTarget.get(head.parentNode)
    if(current_target){
        current_target.classList.add("hidden");
    }

    head.parentNode.querySelectorAll("*").forEach((el)=>{
        el.classList.remove("active")
    });

    head.classList.add("active")

    let target = head.getAttribute("target")
    if(!target){console.error("target is null");return}
    let target_el = document.getElementById(target)
    if(!target_el){console.error("target element not found");return}

    _tabHeadMap_currentTarget.set(head.parentNode,target_el);
    
    target_el.classList.remove("hidden");
}