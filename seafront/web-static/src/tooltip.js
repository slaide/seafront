"use strict";

/** tooltip configuration */
export const tooltipConfig={
    enabled:true,
    delayMs:400
};
/** @type {{currentTooltipElement:HTMLElement|null}} */
const tooltip={
    currentTooltipElement:null,
};

window.addEventListener("keydown",(ev)=>{
    if(ev.key=="F1"){
        // always prevent F1 input because its utterly useless
        ev.preventDefault();

        // check for an element to accept the input
        const el=activeElement.el;
        if(!el)return;

        // activate element tooltip
        elementShowtooltip.get(el)?.(el);
    }
});
/** @type {{el?:HTMLElement}} */
let activeElement={el:undefined};
/** @type {WeakMap<HTMLElement,function(HTMLElement):void>} */
const elementShowtooltip=new WeakMap();
/** @type {WeakSet<HTMLElement>} */
const elementStopTooltipRegistered=new WeakSet();

/**
 * @param {HTMLElement} el
 */
export function enabletooltip(el) {
    /** timer to wait a short while between hover start and tooltip popup @type {number?} */
    let timer = null;
    /** @type {MutationObserver?} */
    let observer = null;

    el.addEventListener("mouseenter", ()=>starttimer());

    // allow pressing f1 while element is focused or mouse is hovered
    // to manually open tooltip, even when tooltips are disabled.
    let focuscount=0;
    el.addEventListener("focus",(ev)=>{
        const el=ev.currentTarget;
        if(!(el instanceof HTMLElement))return;
        focuscount++;
        if(focuscount==1){
            activeElement.el=el;
        }
    })
    el.addEventListener("mouseenter",(ev)=>{
        const el=ev.currentTarget;
        if(!(el instanceof HTMLElement))return;
        focuscount++;
        if(focuscount==1){
            activeElement.el=el;
        }
    })
    el.addEventListener("mouseleave",(ev)=>{
        const el=ev.currentTarget;
        focuscount--;
        if(focuscount==0){
            activeElement.el=undefined;
        }
    })
    el.addEventListener("blur",(ev)=>{
        const el=ev.currentTarget;
        focuscount--;
        if(focuscount==0){
            activeElement.el=undefined;
        }
    })
    elementShowtooltip.set(el,showtooltip);

    return;

    /**
     * create and display tooltip element
     * @param {HTMLElement?} overrideEl
     */
    function showtooltip(overrideEl) {
        const targetel=overrideEl??el;

        // if there is already an active element, ignore this
        if(tooltip.currentTooltipElement)return;

        if(!elementStopTooltipRegistered.has(targetel)){
            elementStopTooltipRegistered.add(targetel);
            targetel.addEventListener("mouseleave", stoptooltip);
        }
        
        tooltip.currentTooltipElement = document.createElement("div");
        setTooltip();
        tooltip.currentTooltipElement.classList.add("tooltip");
        document.body.appendChild(tooltip.currentTooltipElement);

        // bounding box of element it references
        const elbb = targetel.getBoundingClientRect();
        // tooltip bounding box ( to measure width, height, and viewport edge clearance )
        const ttbb = tooltip.currentTooltipElement.getBoundingClientRect();

        // initial position: top center of referenced element.
        let left = elbb.left + elbb.width / 2 - ttbb.width / 2;
        let top = elbb.top - ttbb.height;

        // if tooltip overflows out of the viewport to the right:
        // push back inside (to touch right border)
        const right_overflow = window.innerWidth - (left + ttbb.width);
        if (right_overflow < 0) {
            left += right_overflow;
        }
        // if tooltip overflows out of the viewport to the left:
        // push back inside (to touch left border)
        left=Math.max(0,left);
        // if tooltip overflows out of the viewport to the top:
        // push back inside (to touch top border)
        if(top<0){
            top=elbb.bottom;
        }
        // bottom overflow is not possible, since the tooltip is positioned at
        // the top center of the element it references.

        tooltip.currentTooltipElement.style = `left: ${left}px; top: ${top}px;`

        // update tooltip if it changes while popup is active
        // (tooltip can be reactive attribute, so this is a valid use case)
        observer = new MutationObserver((mutationList, obs) => {
            for (const mutation of mutationList) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'tooltip') {
                    setTooltip();
                }
            }
        });
        observer.observe(targetel, {
            attributes: true,               // watch for attribute changes
            attributeFilter: ['tooltip'],   // only fire for this attribute
            attributeOldValue: true         // include oldValue in the MutationRecord
        });

        return;

        /**
         * set tooltip to current value of tooltip attribute on element
         */
        function setTooltip(){
            if(tooltip.currentTooltipElement){
                const tooltipText = targetel.getAttribute("tooltip") ?? "";
                // Process tooltip text: trim leading whitespace and convert newlines to <br> tags
                const tooltipHtml = tooltipText.trimStart().replace(/\n/g, "<br>");
                tooltip.currentTooltipElement.innerHTML = tooltipHtml;
            }
        }
    }
    /** stop tooltip display */
    function stoptooltip() {
        const targetel=el;

        elementStopTooltipRegistered.delete(targetel);
        targetel.removeEventListener("mouseleave", stoptooltip);

        if(tooltip.currentTooltipElement){
            tooltip.currentTooltipElement.parentElement?.removeChild(tooltip.currentTooltipElement);
            tooltip.currentTooltipElement = null;
        }

        if(observer){
            observer.disconnect();
            observer=null;
        }

        if(timer){
            clearTimeout(timer);
            timer = null;
        }
    }
    /**
     * initiate waiting to display tooltip timer
     * @param {HTMLElement?} [newel=undefined]
     */
    function starttimer(newel) {
        // if there is already an active element, ignore this
        if(tooltip.currentTooltipElement)return;

        if(!tooltipConfig.enabled)return;
        const targetel=newel??el;

        elementStopTooltipRegistered.add(targetel);
        targetel.addEventListener("mouseleave", stoptooltip);
        timer = setTimeout(showtooltip, tooltipConfig.delayMs);
    }
}
