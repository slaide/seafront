/**
 * 
 * @param {HTMLElement} element 
 */
p.bare_init_tooltip=function(element){
    /// init tooltip to be definitely visible (which it should be, in the top left corner of the viewport)
    element.style.left="0px"
    element.style.top="0px"
}
/**
 * @param {HTMLElement} element
 */
p.init_tooltip=function(element){
    /// actually init element position to top center of the element that it is attached to
    let rect=element.element_anker.getBoundingClientRect()

    let left_offset=rect.left+rect.width/2
    let top_offset=rect.top
    
    element.style.left=left_offset+"px"
    element.style.top=top_offset+"px"

    // if tooltip is outside the viewport, shove it back in
    let tooltip_rect=element.getBoundingClientRect()

    let top_min=10
    let bottom_min=10
    let left_min=15
    let right_min=15

    // check left
    if(tooltip_rect.left<0){
        element.style.left=left_min+"px"
    }
    // check right
    let tooltip_right_offset=tooltip_rect.right-window.innerWidth
    if(tooltip_right_offset>0){
        element.style.left=(left_offset-tooltip_right_offset-right_min)+"px"
    }
    // check top
    let tooltip_top_offset=tooltip_rect.top-top_min
    if(tooltip_top_offset<0){
        console.log("top offset too small",tooltip_rect.top)
        element.style.top=top_offset-tooltip_top_offset+"px"
    }
    // check bottom
    let tooltip_bottom_offset=tooltip_rect.bottom-window.innerHeight
    if(tooltip_bottom_offset>0){
        element.style.top=(top_offset-tooltip_bottom_offset)+"px"
    }
}
