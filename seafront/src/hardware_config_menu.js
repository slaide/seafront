/**
 * 
 * @param {ConfigItem} item 
 * @returns {"number"|"text"}
 */
function getInputType(item) {
    if (item.value_kind == "float") {
        return "number"
    } else if (item.value_kind == "int") {
        return "text"
    } else if (item.value_kind == "text") {
        return "text"
    } else {
        throw new Error("unknown type " + item.value_kind)
    }
}
/**
 * 
 * @param {ConfigItem} item 
 * @returns 
 */
function itemIsRawInput(item) {
    return item.value_kind == "float" || item.value_kind == "int" || item.value_kind == "text"
}
/**
 * 
 * @param {ConfigItem} item 
 * @returns 
 */
function itemIsOptionInput(item) {
    return item.value_kind == "option"
}
/**
 * 
 * @param {ConfigItem} item 
 * @returns 
 */
function itemIsAction(item) {
    return item.value_kind == "action"
}
/**
 * 
 * @param {ConfigItem} item 
 * @returns 
 */
function executeActionItem(item) {
    if (!itemIsAction(item)) { throw new Error("item is not an action") }

    const data = {}

    try {
        progress_indicator.run("machine action " + item.name)
    } catch (e) {
        message_open("error", "cannot currently execute action", e)
        return
    }

    // send xhr with item.value as url
    new XHR(true)
        .onload((xhr) => {
            progress_indicator.stop()

            let response = JSON.parse(xhr.responseText)
        })
        .onerror((xhr) => {
            progress_indicator.stop()

            console.error("action failed", item, xhr.responseText)
        })
        // @ts-ignore
        .send(item.value, data, "POST")
}

let machine_value_filter = _p.manage({ value: "" })
/**
 * check if name contains the filter query
 * @param {string} name
 * @returns {boolean}
 */
function match_name(name) {
    if (machine_value_filter.value.length == 0)
        return true;

    const lowercasename = name.toLowerCase()
    const lowercasemachine_value_filter = machine_value_filter.value.toLowerCase()
    const ret = lowercasename.includes(lowercasemachine_value_filter)
    return ret
}
/** @type {ConfigItem[]} */
let filtered_machine_defaults = _p.manage([])

/**
 * apply the filter to the machine_defaults and store the result in filtered_machine_defaults
 * 
 * used as callback (on the search bar)
 */
function filter_results() {
    filtered_machine_defaults.length = 0
    filtered_machine_defaults.splice(0, 0, ...(microscope_config.machine_config || []).filter((item) => match_name(item.name)))
}