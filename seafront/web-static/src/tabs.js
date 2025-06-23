"use strict";

/** @typedef {HTMLElement&{tabs?:TabEntryElement[],activeTab?:TabEntryElement}} TabContainerElement */
/** @typedef {HTMLElement&{header:HTMLElement}} TabEntryElement */

/**
 * @param {TabContainerElement} el
 */
export function initTabs(el) {
    el.tabs = []
    el.classList.add("tabs")

    /**
     * make the tab associated with el (which is the corresponding tab bar entry) the active tab 
     * @param {TabEntryElement} el 
     */
    function activateTab(el) {
        /** @ts-ignore @type {TabContainerElement} */
        const parent=el.parentElement;

        // hide previous element
        const activetab=parent.activeTab;
        if(activetab){
            activetab.classList.add("hidden");
            activetab.header.classList.remove("active");
        }
        // set new element as active
        parent.activeTab = el
        el.classList.remove("hidden")

        el.header.classList.add("active")
    }

    for (let child of el.children) {
        // skip elements that are not visual
        if (["style", "script"].indexOf(child.tagName.toLowerCase()) != -1) continue;

        /// @ts-ignore
        el.tabs.push(child)
        child.classList.add("tab")
        child.classList.add("hidden")
    }

    let tabbar = document.createElement("div")
    tabbar.classList.add("tabbar")
    let child_i = -1
    for (let child of el.tabs) {
        child_i++

        let tabheader = document.createElement("div")
        child.header = tabheader

        tabheader.classList.add("tabheader")
        let tabname = child.getAttribute("tabname")
        if (!tabname) {
            console.warn("Tab has no tabname", child_i, child)
            tabname = `Tab ${child_i}`
        }
        tabheader.innerText = tabname
        tabbar.appendChild(tabheader)

        tabheader.addEventListener("click", () => activateTab(child))
    }

    // activate first tab
    /** @ts-ignore @type {TabEntryElement} */
    const firstchild=tabbar.children[0];
    firstchild?.click();

    if (el.tabs.length == 0) {
        el.appendChild(tabbar)
    } else {
        el.insertBefore(tabbar, el.tabs[0])
    }
}
