"use strict";

/**
 * @typedef {{filename:string,comment:string,timestamp:string,cell_line:string,plate_type:string}} ConfigFileInfo
 */

/** @type {ConfigFileInfo[]} */
let files = _p.manage([])

/**
 * @param {ConfigFileInfo} file_data
 */
function load_remote_config(file_data) {
    let data = { config_file: file_data.filename }

    new XHR(false)
        .onload((xhr) => {
            const response = JSON.parse(xhr.responseText)

            microscopeConfigOverride(response.file)
        })
        .onerror((xhr) => {
            message_open("error", "failed to load config", xhr.responseText)
        })
        .send("/api/acquisition/config_fetch", data, "POST")
}

/** @type {HTMLElement?} */
let config_load_modal_element = null
/** @type {HTMLElement?} */
let config_store_modal_element = null

function config_store() {
    if (config_store_modal_element == null) {
        config_store_modal_element = document.getElementById("config-store-modal")
        if (config_store_modal_element == null) { throw new Error("config-store-modal not found") }
        if (config_store_modal_element.parentElement == null) { throw new Error("bug") }
        config_store_modal_element.parentElement.removeChild(config_store_modal_element)
        config_store_modal_element.classList.remove("hidden")
    }

    if (config_store_modal_element == null) { throw new Error("unreachable") }

    spawnModal(
        "save config file",
        config_store_modal_element,
        {
            buttons:
                [
                    {
                        title: "save",
                        onclick: () => {
                            const config_store_modal_filename_element = document.getElementById("config-store-modal-filename")
                            const config_store_modal_comment_element = document.getElementById("config-store-modal-comment")

                            if (!(config_store_modal_filename_element instanceof HTMLInputElement)) { throw new Error("") }
                            const filename = config_store_modal_filename_element.value
                            if (!(config_store_modal_comment_element instanceof HTMLTextAreaElement)) { throw new Error("") }
                            const comment = config_store_modal_comment_element.value

                            const microscope_config_copy = structuredClone(_p.getUnmanaged(microscope_config))
                            // remove the wells that are just used for display purposes
                            microscope_config_copy.plate_wells = microscope_config_copy.plate_wells.filter(w => w.col > 0 && w.row > 0)
                            // and adjust the indices to be 0-indexed again (they are 1-indexed for displaying)
                            microscope_config_copy.plate_wells.forEach(w => { w.row--; w.col-- })

                            const data = {
                                "config_file": microscope_config_copy,
                                "filename": filename,
                                "comment": comment,
                            }

                            new XHR(false)
                                .onload((xhr) => {
                                    const response = JSON.parse(xhr.responseText)

                                    config_store_modal_filename_element.value = ""
                                    config_store_modal_comment_element.value = ""

                                    modal_close()
                                })
                                .onerror((xhr) => {
                                    message_open("error", "failed to store config", xhr.responseText)
                                })
                                .send("/api/acquisition/config_store", data, "POST")
                        }
                    }
                ]
        }
    )
}
function config_list() {
    const data = {}
    new XHR(false)
        .onload((xhr) => {
            const response = JSON.parse(xhr.responseText)

            files.length = 0
            files.splice(0, 0, ...response.configs)

            if (config_load_modal_element == null) {
                config_load_modal_element = document.getElementById("config-load-modal")

                if (config_load_modal_element == null) { throw new Error() }
                if (config_load_modal_element.parentElement == null) { throw new Error() }
                config_load_modal_element.parentElement.removeChild(config_load_modal_element)
                config_load_modal_element.removeAttribute("style")
            }

            spawnModal("Load Configuration File", config_load_modal_element)
        })
        .onerror((xhr) => {
            message_open("error", "failed to retrieve config list", xhr.responseText)
        })
        .send("/api/acquisition/config_list", data, "POST")
}