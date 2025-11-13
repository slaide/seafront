"use strict";

// three.js with webgpu backend:
// "three": "https://cdn.jsdelivr.net/npm/three@0.175.0/build/three.webgpu.min.js",
// three.js with Webgl backend:
// "three": "https://cdn.jsdelivr.net/npm/three@0.175.0/build/three.module.min.js",

// import some stuff, then make it available in the html code

import * as THREE from 'three';

import {
    PlateNavigator,
    makeWellName,
    matWellColor,
    matSiteColor,
    matFovColor,
    matForbiddenAreaColor,
    calculateSitePosition,
    calculateSitePositionPython,
} from "platenavigator";
// these colors are used to create a legend in html
Object.assign(window, { matWellColor, matSiteColor, matFovColor, matForbiddenAreaColor });

import { registerNumberInput } from "numberinput";
Object.assign(window, { registerNumberInput });

import { registerFilterSelect } from "filterselect";
Object.assign(window, { registerFilterSelect });

import { ChannelImageView } from "channelview";

import {
    parseConfigNamespaces,
    flattenNamespaceTree,
    toggleNamespaceExpansion,
    expandAllNamespaces,
    collapseAllNamespaces,
} from "./namespace-parser.js";

import { APIClient, WebSocketManager } from "./api-client.js";
import { MockAPIClient, MockWebSocketManager } from "./mock-api-client.js";

Object.assign(window,{toggleNamespaceExpansion});

const placeholder_microscope_name="<undefined>";

import { tooltipConfig, enabletooltip } from "tooltip";

/**
 *  Interface settings with defaults
 * @type {CachedInterfaceSettings}
 */
const defaultInterfaceSettings = {
    tooltip: {
        enabled: true,
        delayMs: 400
    },
    busyLingerMs:0,
    theme:"light",
    savedAt:"0",
    popupHideDelayMs: 500
};

// Current interface settings (will be merged with saved settings)
const interfaceSettings = structuredClone(defaultInterfaceSettings);

Object.assign(window, { enabletooltip, tooltipConfig, interfaceSettings });

import { initTabs } from "tabs";
Object.assign(window, { initTabs });

import { makeHistogram, histogramLayout, histogramConfig } from "histogram";

/**
 * add 'disabled' attribute to an element if condition is true, otherwise removes the attribute.
 * @param {HTMLElement} el
 * @param {function():boolean} condition
 */
function disableElement(el, condition) {
    if (condition()) {
        el.setAttribute("disabled", "true");
    } else {
        el.removeAttribute("disabled");
    }
}
Object.assign(window, { disableElement });

// import alpine, and start
import { Alpine } from "alpine";
import JSON5 from "json5";
const json5=JSON5;
window.addEventListener("load", () => {
    Alpine.start();
});

/**
 * clone an object.
 *
 * attempts structuredClone first, with fallback to json round trip.
 * @template T
 * @param {T} o
 * @returns {T}
 */
function cloneObject(o) {
    try {
        return structuredClone(o);
    } catch (e) {
        return json5.parse(JSON.stringify(o));
    }
}

document.addEventListener("alpine:init", () => {
    Alpine.data("microscope_state", () => ({
        /** url to microscope (hardware) server (default to same origin as gui) */
        server_url: window.location.origin,

        // These will be initialized in the init() method
        /** @type {APIClient?}  */
        _api: null,
        get api(){
            if(!this._api)throw new Error(`this._api is null`);
            return this._api;
        },
        /** @type {WebSocketManager?} */
        _wsManager: null,
        get wsManager(){
            if(!this._wsManager)throw new Error(`this._wsManager is null`);
            return this._wsManager;
        },

        tooltipConfig,
        interfaceSettings,

        // Central error modal state
        /** @type {{ show: boolean, title: string, message: string, timestamp: Date | null, showReloadButton: boolean }} */
        centralError: {
            show: false,
            title: '',
            message: '',
            timestamp: null,
            showReloadButton: false
        },

        // Mock mode indicator
        mockModeActive: false,

        // Cached microscopes list for Settings panel
        /** @type {string[]} */
        cachedMicroscopes: [],

        /** @type {HardwareLimits|null} */
        limits: null,

        themes: ["light", "dark"],

        /** @type {{x: number, y: number}|null} */
        plateCursorPosition: null,
        theme: "light", // Will be loaded from microscope-specific cache after init
        
        /** Selection mode for plate navigator: 'wells' or 'sites' */
        plateSelectionMode: "wells",

        // Busy indicator state
        showBusyIndicator: false,
        busyIndicatorHiding: false,
        /** @type {number|null} */
        busyIndicatorTimeout: null,
        busyLingerMs: 500, // Default value, will be updated by loadInterfaceSettings()

        get busyIndicatorClass() {
            return {
                'busy-indicator': this.showBusyIndicator && !this.busyIndicatorHiding,
                'hide': this.busyIndicatorHiding
            };
        },

        // Central error display method
        /**
         * @param {string} title
         * @param {string} message
         */
        showError(title, message) {
            console.log(`showError called: ${title} - ${message}`);
            this.centralError.title = title;
            this.centralError.message = message;
            this.centralError.timestamp = new Date();
            this.centralError.show = true;
            // Reset reload button flag for non-microscope warnings
            if (title !== 'Microscope Configuration Mismatch') {
                this.centralError.showReloadButton = false;
            }
        },

        /**
         * Show mock mode indicator banner
         */
        showMockModeIndicator() {
            this.mockModeActive = true;
        },

        /**
         * Handle busy indicator with 1.5s delay after becoming not busy
         * @param {boolean} isBusy
         */
        handleBusyIndicator(isBusy) {
            if (isBusy) {
                // Microscope is busy - show indicator immediately
                if (this.busyIndicatorTimeout) {
                    clearTimeout(this.busyIndicatorTimeout);
                    this.busyIndicatorTimeout = null;
                }
                this.busyIndicatorHiding = false;
                this.showBusyIndicator = true;
            } else {
                // Microscope is not busy - hide after 1.5s delay
                if (this.showBusyIndicator && !this.busyIndicatorTimeout) {
                    this.busyIndicatorTimeout = setTimeout(() => {
                        this.busyIndicatorHiding = true;
                        // Hide completely after animation finishes
                        setTimeout(() => {
                            this.showBusyIndicator = false;
                            this.busyIndicatorHiding = false;
                            this.busyIndicatorTimeout = null;
                        }, 200); // Match the CSS animation duration
                    }, this.busyLingerMs);
                }
            }
        },

        // Computed property for formatted timestamp in 24-hour format
        get formattedErrorTimestamp() {
            if (!this.centralError.timestamp) return '';
            return this.centralError.timestamp.toLocaleString('en-US', { hour12: false });
        },

        /**
         * @template T
         * @template {object & {detail?:string,message?:string}} E
         * @param {Response} v
         * @param {string} context
         * @param {boolean} showError
         * @returns {Promise<T>}
         */
        async checkMapSquidRequest(v, context = 'API request', showError = true) {
            if (!v.ok) {
                if (v.status == 500) {
                    // Handle internal server errors
                    /** @type {InternalErrorModel} */
                    const error_body = await v.json();

                    const error = `${context} failed with ${v.statusText} ${v.status} because: ${JSON.stringify(error_body)}`;
                    console.error(error);

                    if (showError) {
                        this.showError(context + ' Error', error_body.detail);
                    }
                    throw new Error(error);
                } else if (v.status == 409) {
                    // Handle conflict errors (e.g., microscope busy)
                    /** @type {MicroscopeBusyError} */
                    const error_body = await v.json();

                    const reasons = error_body.detail.busy_reasons.map(r => `  • ${r}`).join('\n');
                    const userMessage = `${error_body.detail.message}:\n${reasons}`;

                    const error = `${context} failed with ${v.statusText} ${v.status} because: ${userMessage}`;
                    console.error(error);

                    if (showError) {
                        this.showError(context + ' Error', userMessage);
                    }
                    throw new Error(error);
                } else {
                    const responseblob=await v.blob();
                    const responsetext=await responseblob.text();
                    try{
                        const repsonsejson=JSON.parse(responsetext);
                        console.warn(repsonsejson);
                    }catch(e){}
                    const error = `${context} failed: ${v.status} ${responsetext}`;
                    if (showError) {
                        this.showError(context + ' Error', `HTTP ${v.status}: ${responsetext}`);
                    }
                    throw new Error(error);
                }
            }
            /** @type {Promise<T>} */
            const ret = await v.json();

            return ret;
        },

        /**
         * Helper to call checkMapSquidRequest with this context (legacy method name)
         * @param {*} response
         * @param {*} context
         * @returns
         */
        checkRequest(response, context) {
            console.log(`checkRequest called: ${context}`);
            return this.checkMapSquidRequest(response, context, true);
        },

        /**
         * Helper method to extract microscope name from machine config
         * @returns {string|null} The microscope name or null if not available
         */
        getMicroscopeName() {
            try {
                const microscopeNameItem = this._microscope_config?.machine_config?.find(
                    item => item.handle === "system.microscope_name"
                );
                if (!microscopeNameItem || !microscopeNameItem.value) {
                    return null;
                }
                if (typeof microscopeNameItem.value !== "string") {
                    console.warn("Microscope name value is not a string:", microscopeNameItem.value);
                    return null;
                }
                return microscopeNameItem.value;
            } catch (error) {
                console.warn("Failed to extract microscope name:", error);
                return null;
            }
        },

        /**
         * Load interface settings for the current microscope
         * @returns {string} The loaded theme or "light" as default
         */
        loadInterfaceSettings() {
            const microscopeName = this.getMicroscopeName();
            if (!microscopeName) {
                console.warn("Cannot load interface settings - microscope name not available");
                return "light";
            }

            try {
                const saved = this.getMicroscopeCache().interface_settings;
                if (saved) {
                    // Merge each section separately to handle nested objects
                    if (saved.tooltip) Object.assign(interfaceSettings.tooltip, saved.tooltip);
                    if (saved.popupHideDelayMs !== undefined) interfaceSettings.popupHideDelayMs = saved.popupHideDelayMs;
                    if (saved.busyLingerMs !== undefined) this.busyLingerMs = saved.busyLingerMs;

                    // Apply to tooltipConfig
                    Object.assign(tooltipConfig, interfaceSettings.tooltip);

                    // Return theme if it exists
                    return saved.theme || "light";
                }
            } catch (error) {
                console.warn(`Failed to load interface settings for microscope ${microscopeName}:`, error);
            }
            return "light"; // default theme
        },

        /**
         * Save interface settings for the current microscope
         */
        saveInterfaceSettings() {
            const microscopeName = this.getMicroscopeName();
            if (!microscopeName) {
                console.warn("Cannot save interface settings - microscope name not available");
                return;
            }

            try {
                // Update from current tooltip config
                Object.assign(interfaceSettings.tooltip, tooltipConfig);

                const toSave = {
                    ...interfaceSettings,
                    busyLingerMs: this.busyLingerMs,
                    theme: this.theme,
                    savedAt: new Date().toISOString()
                };

                const cache = this.getMicroscopeCache();
                cache.interface_settings = toSave;
                this.setMicroscopeCache(cache);
            } catch (error) {
                console.warn("Failed to save interface settings to localStorage:", error);
            }
        },

        /**
         * Get the complete cache data object for the current microscope
         * @returns {MicroscopeCacheData} The cache data object with interface_settings and microscope_config
         */
        getMicroscopeCache() {
            const microscopeName = this.getMicroscopeName();
            if (!microscopeName) {
                console.warn("Cannot get cache - microscope name not available");
                return {};
            }

            try {
                const seafrontData = localStorage.getItem("seafrontdata");
                if (!seafrontData) return {};

                const parsed = JSON.parse(seafrontData);
                const cacheData = parsed[microscopeName] || {};

                return {
                    interface_settings: cacheData.interface_settings || null,
                    microscope_config: cacheData.microscope_config || null
                };
            } catch (error) {
                console.warn(`Failed to load cache for microscope ${microscopeName}:`, error);
                return {};
            }
        },

        /**
         * Set the complete cache data object for the current microscope
         * @param {MicroscopeCacheData} cache The cache data object to store
         */
        setMicroscopeCache(cache) {
            const microscopeName = this.getMicroscopeName();
            if (!microscopeName) {
                console.warn("Cannot set cache - microscope name not available");
                return;
            }

            try {
                // Get existing data or create new structure
                const existingData = localStorage.getItem("seafrontdata");
                const seafrontData = existingData ? JSON.parse(existingData) : {};

                // Ensure microscope entry exists
                if (!seafrontData[microscopeName]) {
                    seafrontData[microscopeName] = {};
                }

                if (cache.interface_settings !== undefined) {
                    seafrontData[microscopeName].interface_settings = cache.interface_settings;
                }
                if (cache.microscope_config !== undefined) {
                    seafrontData[microscopeName].microscope_config = cache.microscope_config;
                }

                // Save back to localStorage
                localStorage.setItem("seafrontdata", JSON.stringify(seafrontData));
            } catch (error) {
                console.error(`Failed to save cache for microscope ${microscopeName}:`, error);
            }
        },

        /**
         * Load all cached microscope names from localStorage and populate the list
         */
        loadCachedMicroscopes() {
            try {
                const seafrontData = localStorage.getItem("seafrontdata");
                if (!seafrontData) {
                    this.cachedMicroscopes = [];
                    return;
                }

                const parsed = JSON.parse(seafrontData);
                this.cachedMicroscopes = Object.keys(parsed).sort();
            } catch (error) {
                console.warn("Failed to load cached microscopes:", error);
                this.cachedMicroscopes = [];
            }
        },

        /**
         * Delete a cached microscope configuration from localStorage
         * @param {string} microscopeName - The name of the microscope to delete
         */
        deleteCachedMicroscope(microscopeName) {
            try {
                const seafrontData = localStorage.getItem("seafrontdata");
                if (!seafrontData) {
                    return;
                }

                const parsed = JSON.parse(seafrontData);
                delete parsed[microscopeName];
                localStorage.setItem("seafrontdata", JSON.stringify(parsed));

                // Refresh the list
                this.loadCachedMicroscopes();
                console.log(`Deleted cached configuration for microscope: ${microscopeName}`);
            } catch (error) {
                console.error(`Failed to delete cached microscope ${microscopeName}:`, error);
            }
        },

        changeTheme() {
            // apply theme to document body
            const el = document.body;

            // remove existing theme
            const existing_theme_class = Array.from(el.classList).find((c) =>
                c.startsWith("theme-"),
            );
            if (existing_theme_class !== undefined) {
                el.classList.remove(existing_theme_class);
            }

            // apply new theme
            el.classList.add(`theme-${this.theme}`);
            
            // update channel view theme if it exists
            if (this.view) {
                this.view.updateTheme();
            }
            
            // save theme to microscope-specific interface settings
            this.saveInterfaceSettings();
        },

        /**
         * 
         * @returns 
         */
        saveMicroscopeConfigToStorage() {
            if (this._microscope_config) {
                try {
                    const microscopeName = this.getMicroscopeName();
                    if (!microscopeName) {
                        console.warn("Cannot save config - microscope name not available");
                        return;
                    }

                    /** @type {CachedMicroscopeConfig} */
                    const configToSave = {
                        microscope_config: this.microscope_config,
                        configIsStored: this.configIsStored,
                        savedAt: new Date().toISOString()
                    };

                    const cache = this.getMicroscopeCache();
                    cache.microscope_config = configToSave;
                    this.setMicroscopeCache(cache);
                } catch (error) {
                    console.error("Failed to save microscope config to localStorage:", error);
                }
            }
        },

        /**
         * Load microscope config from storage for the current microscope
         * @returns {CachedMicroscopeConfig|null}
         */
        loadMicroscopeConfigFromStorage() {
            const microscopeName = this.getMicroscopeName();
            if (!microscopeName) {
                console.warn("Cannot load config - microscope name not available");
                return null;
            }

            try {
                const saved = this.getMicroscopeCache().microscope_config;
                if (saved && saved.microscope_config) {
                    return {
                        microscope_config: saved.microscope_config,
                        configIsStored: saved.configIsStored !== undefined ? saved.configIsStored : false,
                        savedAt:"",
                    };
                }
            } catch (error) {
                console.error("Failed to load microscope config from localStorage:", error);
            }
            return null;
        },

        // Debounced config saving to prevent excessive localStorage writes
        /** @type {number|null} */
        _saveTimeout: null,
        
        saveCurrentConfig() {
            // Clear any pending save
            if (this._saveTimeout) {
                clearTimeout(this._saveTimeout);
            }
            
            // Debounce saves by 50ms to batch rapid changes
            this._saveTimeout = setTimeout(() => {
                this.saveMicroscopeConfigToStorage();
                this._saveTimeout = null;
            }, 50);
        },

        // Debug method to clear localStorage for current microscope
        clearSavedConfig() {
            try {
                const microscopeName = this.getMicroscopeName();
                if (!microscopeName) {
                    console.warn("Cannot clear config - microscope name not available");
                    return;
                }
                const seafrontData = localStorage.getItem("seafrontdata");
                if (seafrontData) {
                    const parsed = JSON.parse(seafrontData);
                    if (parsed[microscopeName]) {
                        delete parsed[microscopeName];
                        localStorage.setItem("seafrontdata", JSON.stringify(parsed));
                        console.log(`Cleared saved config for microscope: ${microscopeName}`);
                    } else {
                        console.log(`No saved config found for microscope: ${microscopeName}`);
                    }
                } else {
                    console.log("No seafrontdata found in localStorage");
                }
            } catch (error) {
                console.error("Failed to clear saved config:", error);
            }
        },

        // Debug method to force an immediate save
        forceSave() {
            console.log("🚀 Force saving current config...");
            if (this._saveTimeout) {
                clearTimeout(this._saveTimeout);
                this._saveTimeout = null;
            }
            this.saveMicroscopeConfigToStorage();
        },

        // Set up warning when user tries to leave page with unsaved protocol
        setupUnsavedChangesWarning() {
            window.addEventListener('beforeunload', (event) => {
                // Check if the current protocol is not saved to server
                // configIsStored tracks whether current config matches a saved server protocol
                if (!this.configIsStored) {
                    // Modern browsers ignore the custom message and show their own
                    const message = 'Your current protocol has not been saved. Are you sure you want to leave?';
                    return message; // For modern browsers
                }
            });
        },

        /**
         * @returns {Promise<HardwareCapabilities>}
         */
        async getHardwareCapabilities() {
            return this.api.post('/api/get_features/hardware_capabilities', {}, {
                context: 'Get hardware capabilities'
            });
        },
        /**
         * @returns {Promise<MachineDefaults>}
         */
        async getMachineDefaults() {
            return this.api.post('/api/get_features/machine_defaults', {}, {
                context: 'Get machine defaults'
            });
        },

        /**
         * @returns {Promise<ConfigListResponse>}
         */
        async getConfigList() {
            return this.api.post('/api/acquisition/config_list', {}, {
                context: 'Get config list'
            });
        },

        /**
         * get plate types from server
         * @returns {Promise<{plategroups:WellPlateGroup[],allplates:Wellplate[]}>}
         * */
        async getPlateTypes() {
            let data = await this.getHardwareCapabilities();

            /** @type {{plategroups:WellPlateGroup[],allplates:Wellplate[]}} */
            let plateinfo = { allplates: [], plategroups: [] };

            for (const key in data.wellplate_types) {
                const value = data.wellplate_types[key];

                // make copy of plate type
                /** @type {Wellplate} */
                const newplate = structuredClone(value);

                plateinfo.allplates.push(newplate);

                /** @type {WellPlateGroup|undefined} */
                let plategroup = plateinfo.plategroups.find(
                    (g) =>
                        g.numwells ==
                        newplate.Num_wells_x * newplate.Num_wells_y,
                );
                if (!plategroup) {
                    plategroup = {
                        label: `${newplate.Num_wells_x * newplate.Num_wells_y} well plate`,
                        numwells: newplate.Num_wells_x * newplate.Num_wells_y,
                        plates: [],
                    };
                    plateinfo.plategroups.push(plategroup);
                }
                plategroup.plates.push(newplate);
            }

            // sort by number of wells, in descending order
            plateinfo.plategroups.sort(
                (g1, g2) =>
                    parseInt("" + g1.numwells) - parseInt("" + g2.numwells),
            );

            return plateinfo;
        },

        /**
         * @return {Promise<AcquisitionConfig>}
         **/
        async defaultConfig() {
            // Load default protocol from server using existing config_fetch endpoint
            try {
                const response = await this.api.post('/api/acquisition/config_fetch',
                    { config_file: "default.json" },
                    { context: 'Load default config', showError: false }
                );
                return response.file;
            } catch (error) {
                const errorMsg = `Failed to load default protocol. Please ensure default.json exists and run: uv run python scripts/generate_default_protocol.py`;
                console.error(errorMsg);
                throw new Error(errorMsg);
            }
        },

        /** protocols stored on server @type {ConfigListEntry[]} */
        protocol_list: [],
        protol_list_filters: {
            filename: "",
            project_name: "",
            plate_name: "",
            comment: "",
            cell_line: "",
            plate_type: "",
        },
        get filtered_protocol_list() {
            return this.protocol_list
                .filter((p) => {
                    if (this.protol_list_filters.filename) {
                        return (
                            p.filename
                                .toLowerCase()
                                .indexOf(
                                    this.protol_list_filters.filename.toLowerCase(),
                                ) > -1
                        );
                    }
                    return true;
                })
                .filter((p) => {
                    if (this.protol_list_filters.project_name) {
                        return (
                            p.project_name
                                .toLowerCase()
                                .indexOf(
                                    this.protol_list_filters.project_name.toLowerCase(),
                                ) > -1
                        );
                    }
                    return true;
                })
                .filter((p) => {
                    if (this.protol_list_filters.plate_name) {
                        return (
                            p.plate_name
                                .toLowerCase()
                                .indexOf(
                                    this.protol_list_filters.plate_name.toLowerCase(),
                                ) > -1
                        );
                    }
                    return true;
                })
                .filter((p) => {
                    if (this.protol_list_filters.comment) {
                        return (
                            p.comment
                                .toLowerCase()
                                .indexOf(
                                    this.protol_list_filters.comment.toLowerCase(),
                                ) > -1
                        );
                    }
                    return true;
                })
                .filter((p) => {
                    if (this.protol_list_filters.cell_line) {
                        return (
                            p.cell_line
                                .toLowerCase()
                                .indexOf(
                                    this.protol_list_filters.cell_line.toLowerCase(),
                                ) > -1
                        );
                    }
                    return true;
                })
                .filter((p) => {
                    if (this.protol_list_filters.plate_type) {
                        return (
                            p.plate_type.Model_id.toLowerCase().indexOf(
                                this.protol_list_filters.plate_type.toLowerCase(),
                            ) > -1
                        );
                    }
                    return true;
                });
        },
        async refreshConfigList() {
            this.protocol_list = (await this.getConfigList()).configs;
        },

        /** used in GUI to configure filename when storing current config on server */
        configStore_filename: "",
        configStore_overwrite_on_conflict: false,
        async storeConfig() {
            const configStoreEntry = {
                // structuredClone does not work on this
                config_file: this.microscope_config_copy,

                filename: this.configStore_filename,
                comment: this.microscope_config.comment,
                overwrite_on_conflict: this.configStore_overwrite_on_conflict,
            };
            await this.Actions.storeConfig(configStoreEntry);

            // ensure no config is overwritten by accident afterwards
            this.configStore_overwrite_on_conflict = false;
            this.configStore_filename = "";

            this.configIsStored = true;

            // refresh list after store (to confirm successful store)
            await this.refreshConfigList();
        },
        /**
         *
         * @param {ConfigListEntry} protocol
         */
        async loadConfig(protocol) {
            const newconfig = await this.Actions.loadConfig({
                config_file: protocol.filename,
            });

            // Save the current complete plate_wells grid before replacing
            const currentPlateWells = this.microscope_config.plate_wells;

            // Save server-provided fields that shouldn't be overwritten by the protocol
            const serverProvidedMachineConfig = this.microscope_config.machine_config;
            const serverProvidedChannels = this.microscope_config.channels;

            // Load the new configuration
            Object.assign(this.microscope_config, newconfig.file);

            // Restore server-provided fields (machine_config contains microscope name)
            this.microscope_config.machine_config = serverProvidedMachineConfig;

            // Restore hardware capabilities if the protocol doesn't provide channels
            if (!newconfig.file.channels || newconfig.file.channels.length === 0) {
                this.microscope_config.channels = serverProvidedChannels;
            }
            
            // Merge well selection state from loaded protocol into complete grid
            if (newconfig.file.plate_wells && currentPlateWells) {
                // Reset all wells to unselected first
                currentPlateWells.forEach(well => well.selected = false);
                
                // Apply selection state from loaded protocol
                const loadedWellsMap = new Map();
                newconfig.file.plate_wells.forEach(well => {
                    const key = `${well.col},${well.row}`;
                    loadedWellsMap.set(key, well.selected);
                });
                
                // Update selection state in complete grid
                currentPlateWells.forEach(well => {
                    const key = `${well.col},${well.row}`;
                    if (loadedWellsMap.has(key)) {
                        well.selected = loadedWellsMap.get(key);
                    }
                });
                
                // Restore the complete plate_wells grid with updated selections
                this.microscope_config.plate_wells = currentPlateWells;
                
                // Save config after well selections are loaded
                this.saveCurrentConfig();
            }

            this.configIsStored = true;
        },

        /** indicate if config currently present in interface is stored somewhere */
        configIsStored: false,

        // keep track of global initialization state
        initDone: false,
        async manualInit() {
            await this.initSelf();
            
            // Only recreate well grid if we don't have a complete one (i.e., from server default)
            // If we loaded from localStorage, we should already have a complete grid with selections
            if (this._microscope_config && this._microscope_config.wellplate_type) {
                const currentWells = this._microscope_config.plate_wells || [];
                const expectedWellCount = this._microscope_config.wellplate_type.Num_wells_x * this._microscope_config.wellplate_type.Num_wells_y;
                
                
                // Only recreate wells if we have fewer wells than expected (sparse data from server)
                if (currentWells.length < expectedWellCount) {
                    const completeWells = this.createPlateWells(this._microscope_config.wellplate_type);

                    // Merge selection state from sparse data into complete grid
                    if (currentWells.length > 0) {
                        const wellSelectionMap = new Map();
                        currentWells.forEach(well => {
                            const key = `${well.col},${well.row}`;
                            wellSelectionMap.set(key, well.selected);
                        });

                        completeWells.forEach(well => {
                            const key = `${well.col},${well.row}`;
                            if (wellSelectionMap.has(key)) {
                                well.selected = wellSelectionMap.get(key);
                            }
                        });
                    }

                    this._microscope_config.plate_wells = completeWells;

                    // Save config after well selections are merged during initialization
                    this.saveCurrentConfig();
                }
            }
            
            this.initDone = true;

            // Set up beforeunload warning for unsaved changes
            this.setupUnsavedChangesWarning();


            await this.refreshConfigList();

            // Restore the configIsStored status AFTER all initialization is complete
            if (this._savedConfigIsStored !== undefined) {
                this.configIsStored = this._savedConfigIsStored;
                this._savedConfigIsStored = undefined; // Clean up
            }
            
            // Restore site selections AFTER all grid initialization is complete
            if (this._savedSiteSelections !== undefined) {
                // Restore saved site selections by updating existing mask entries
                for (const savedSite of this._savedSiteSelections) {
                    const existingSite = this.microscope_config.grid.mask.find(
                        s => s.col === savedSite.col && s.row === savedSite.row
                    );
                    if (existingSite) {
                        existingSite.selected = savedSite.selected;
                    }
                }
                
                this._savedSiteSelections = undefined; // Clean up
                
                // Update the visual display to show restored selections
                if (this.plateNavigator) {
                    this.plateNavigator.refreshWellColors(this.microscope_config);
                }
            }
        },
        /** @type {boolean|undefined} */
        _savedConfigIsStored:undefined,
        
        /** @type {AcquisitionWellSiteConfigurationSiteSelectionItem[]|undefined} */
        _savedSiteSelections: undefined,

        /**
         *
         * @param {string} name
         * @returns {MachineConfigItem|null}
         */
        getMachineConfigItem(name) {
            for (const configitem of this.microscope_config.machine_config) {
                if (configitem.handle == name) {
                    return configitem;
                }
            }
            return null;
        },

        /**
         *
         * @param {MachineConfigItem} config
         * @returns {Promise<BasicSuccessResponse>}
         */
        async runMachineConfigAction(config) {
            if (config.value_kind != "action") {
                throw new Error(`cannot runMachineConfigAction on non-action config ${config.handle} (kind=${config.value_kind})`);
            }

            console.log(`executing action: '${config.value}'`);
            return this.api.post(config.value, {}, {
                context: `Execute action: ${config.handle}`
            });
        },

        // keep track of number of open websockets (to limit frontend load)
        _numOpenWebsockets: 0,
        
        /** @typedef {{resolve: Function, reject: Function, channel_info:ChannelInfo}} PendingImageRequest */
        /** @type {Map<string, PendingImageRequest>} */
        pending_image_requests: new Map(),
        /** @type {{channel_handle:string,request:PendingImageRequest,metadata:any}|null} */
        _image_current_request: null,

        initImageWebSocket() {
            // Check if already connected via WebSocketManager
            if (this.wsManager.isConnected('image')) {
                return;
            }

            try {
                this.wsManager.createConnection('image', '/ws/get_info/acquired_image', {
                    onOpen: (ws) => {
                        console.log('Image WebSocket connected');
                        // Set initial binary type for metadata (string)
                        ws.binaryType = "blob";
                    },
                    onMessage: (ev) => {
                        // The protocol sends metadata first (string), then image data (ArrayBuffer)
                        if (typeof ev.data === 'string') {
                            // Metadata message
                            const metadata = json5.parse(ev.data);
                            // Find the matching request (FIFO queue)
                            const nextEntry = this.pending_image_requests.entries().next().value;
                            if (nextEntry) {
                                const [channel_handle, request] = nextEntry;
                                if (request) {
                                    this._image_current_request = { channel_handle, request, metadata };
                                    // Switch to receive ArrayBuffer for image data
                                    const ws = this.wsManager.getConnection('image');
                                    if (ws) ws.binaryType = "arraybuffer";
                                }
                            }
                        } else {
                            // Image data (ArrayBuffer)
                            if (this._image_current_request) {
                                const img_data = ev.data;

                                /** @type {CachedChannelImage} */
                                const img = Object.assign(this._image_current_request.metadata, {
                                    data: img_data,
                                    info: this._image_current_request.request.channel_info,
                                });

                                this._image_current_request.request.resolve(img);
                                this.pending_image_requests.delete(this._image_current_request.channel_handle);
                                this._image_current_request = null;

                                // Switch back to blob for next metadata
                                const ws = this.wsManager.getConnection('image');
                                if (ws) ws.binaryType = "blob";
                            }
                        }
                    },
                    onError: (event) => {
                        console.warn('Image WebSocket error:', event);
                        // Reject all pending requests
                        this.pending_image_requests.forEach(request => {
                            request.reject(new Error('WebSocket error'));
                        });
                        this.pending_image_requests.clear();
                        this._image_current_request = null;
                    },
                    onClose: () => {
                        console.log('Image WebSocket closed');
                        // Reject all pending requests
                        this.pending_image_requests.forEach(request => {
                            request.reject(new Error('WebSocket closed'));
                        });
                        this.pending_image_requests.clear();
                        this._image_current_request = null;
                    },
                    autoReconnect: true,
                });
            } catch (error) {
                console.warn('Failed to create image WebSocket:', error);
            }
        },
        
        /**
         *
         * @param {ChannelInfo} channel_info
         * @param {number} [downsample_factor=1]
         * @returns
         */
        async fetch_image(channel_info, downsample_factor = 1) {
            // Initialize persistent WebSocket if needed
            this.initImageWebSocket();

            // Wait for WebSocket to be ready
            const isReady = await this.wsManager.waitForConnection('image', 5000);
            if (!isReady) {
                throw new Error('Image WebSocket not available');
            }

            const channel_handle = channel_info.channel.handle;

            /**@type {Promise<CachedChannelImage>}*/
            const finished = new Promise((resolve, reject) => {
                // Store the request
                this.pending_image_requests.set(channel_handle, {
                    resolve,
                    reject,
                    channel_info,
                });

                // Send the request through WebSocketManager
                this.wsManager.send('image', channel_handle);
                this.wsManager.send('image', `${downsample_factor}`);
            });

            const data = await finished;
            // console.log("fetched image",data.info.channel.name, data);
            return data;
        },
        /**
         *
         * @param {CoreCurrentState} data
         */
        async updateMicroscopeStatus(data) {
            const timestamps = [];

            if(data.latest_imgs){
                const img_keys=Object.keys(data.latest_imgs);
                for (let channelhandle of img_keys) {
                    const channel = data.latest_imgs[channelhandle];
                    timestamps.push(channel.timestamp);
                }
            }
            // console.log(`updateMicroscopeStatus with`, timestamps)

            // Check for microscope name changes during reconnection
            if (data.microscope_name && this.state && this.state.microscope_name) {
                const previousMicroscopeName = this.state.microscope_name;
                const currentMicroscopeName = data.microscope_name;

                if (previousMicroscopeName !== currentMicroscopeName && previousMicroscopeName!==placeholder_microscope_name) {
                    console.warn(`Microscope name changed from '${previousMicroscopeName}' to '${currentMicroscopeName}'`);
                    this.showMicroscopeNameChangeWarning(previousMicroscopeName, currentMicroscopeName);
                }
            }

            // update state with data from 'data' object
            this.state = data;

            // Handle busy indicator with delay
            this.handleBusyIndicator(data.is_busy);

            // Sync streaming state from server
            if (data.is_streaming !== undefined) {
                this.isStreaming = data.is_streaming;
            }
            
            // Update acquisition status WebSocket based on acquisition state
            if (data.current_acquisition_id) {
                // Start acquisition status WebSocket if not already running
                if (!this.wsManager.isConnected('acquisition_status')) {
                    this.startAcquisitionStatusWebSocket(data.current_acquisition_id);
                }
            } else {
                // No active acquisition, close WebSocket and clear status
                if (this.wsManager.getConnection('acquisition_status')) {
                    this.wsManager.closeConnection('acquisition_status');
                    this.acquisition_status_ws = null;
                }
                this.latest_acquisition_status = null;
            }
            
            // Check for acquisition errors and display them
            if (data.last_acquisition_error && data.last_acquisition_error_timestamp !== this._last_displayed_acquisition_error_timestamp) {
                // Format timestamp for user-friendly display in local time
                let displayMessage = data.last_acquisition_error;
                if (data.last_acquisition_error_timestamp) {
                    const localTime = new Date(data.last_acquisition_error_timestamp).toLocaleTimeString('en-US', { hour12: false });
                    displayMessage = `[${localTime}] ${data.last_acquisition_error}`;
                }
                this.showError('Acquisition Failed', displayMessage);
                this._last_displayed_acquisition_error_timestamp = data.last_acquisition_error_timestamp;
            }

            if (this._numOpenWebsockets < 1 && this.state.latest_imgs != null) {
                for (const [channel_handle, channel_info] of Object.entries(
                    this.state.latest_imgs,
                )) {
                    // ignore laser autofocus image (which is not actually useful for anything other than debugging, for which it has its own button)
                    if (channel_handle == "laser_autofocus") continue;

                    const cached_image =
                        this.cached_channel_image.get(channel_handle);

                    const image_cache_outdated =
                        /*no image in cache*/ cached_image == null ||
                        /* cached image older than latest image */ channel_info.timestamp >
                        cached_image.info.timestamp;

                    //console.log(`${channel_handle} image_cache_outdated? ${image_cache_outdated} (${channel_info.timestamp} ${cached_image?.info.timestamp})`)
                    if (!image_cache_outdated) {
                        continue;
                    }

                    const img = await this.fetch_image(channel_info, 2);
                    this.cached_channel_image.set(channel_handle, img);
                }
            }

            // Update objective position if PlateNavigator is available and position data exists
            if (this.plateNavigator && this.state.adapter_state && this.state.adapter_state.stage_position) {
                const position = this.state.adapter_state.stage_position;
                this.plateNavigator.updateObjectivePosition(position.x_pos_mm, position.y_pos_mm);
            }
        },

        /** @type {PlateNavigator|null} */
        plateNavigator: null,
        /**
         *
         * @param {HTMLElement} el
         */
        initPlateNavigator(el) {
            this.plateNavigator = new PlateNavigator(el, this);

            this.plateNavigator.cameraFit({
                ax: 0,
                ay: 0,
                bx: this.microscope_config.wellplate_type.Length_mm,
                by: this.microscope_config.wellplate_type.Width_mm,
            });

            // Set up double-click callback to move objective
            this.plateNavigator.setObjectiveMoveCallback(async (x_mm, y_mm) => {
                await this.moveObjectiveTo(x_mm, y_mm);
            });

            // Set up shift+drag callback to select wells or sites
            this.plateNavigator.setWellSelectionCallback(
                /**
                 * @type {function(string[],string,{
                 *          minX: number,maxX: number,minY: number,maxY: number
                 * }):Promise<void>}
                 * */
                async (wellNames, mode, selectionBounds) => {
                    if (this.plateSelectionMode === "wells") {
                        await this.selectWellsByNames(wellNames, mode);
                    } else if (this.plateSelectionMode === "sites") {
                        await this.selectSitesByWellArea(wellNames, mode, selectionBounds);
                    }
                }
            );
        },

        /**
         * Move objective to specified plate coordinates
         * @param {number} x_mm - X coordinate in mm
         * @param {number} y_mm - Y coordinate in mm (backend/physical coordinates, origin top-left at A1)
         */
        async moveObjectiveTo(x_mm, y_mm) {
            try {
                // PlateNavigator returns backend/physical coordinates (origin top-left, A1 at top-left)
                // Pass them directly to the backend API which expects physical coordinates
                const moveRequest = {
                    x_mm: x_mm,
                    y_mm: y_mm
                };

                await this.Actions.moveTo(moveRequest);
            } catch (error) {
                console.error("Failed to move objective:", error);
                throw error;
            }
        },

        /**
         * Trigger pulse animation on the objective FOV marker
         * Scales from 15x to normal size over 10 seconds for easy visibility
         */
        pulseObjectiveFov() {
            if (this.plateNavigator) {
                this.plateNavigator.pulseObjectiveFov();
            }
        },

        /**
         * Select or deselect wells by their names (e.g., ["A01", "A02", "B01"])
         * @param {string[]} wellNames - Array of well names to select/deselect
         * @param {string} mode - 'select' to add wells to selection, 'deselect' to remove wells from selection
         */
        async selectWellsByNames(wellNames, mode = 'select') {
            if (!this._microscope_config || !this._microscope_config.plate_wells) {
                console.warn("Cannot select wells - no plate configuration loaded");
                return;
            }

            // Create a Set for fast lookup
            const wellNamesSet = new Set(wellNames);
            
            // Update selection state only for the specified wells
            let changedCount = 0;
            for (const well of this._microscope_config.plate_wells) {
                // Skip header wells (col < 0 or row < 0)
                if (well.col < 0 || well.row < 0) continue;
                
                const wellName = makeWellName(well.col, well.row);
                if (wellNamesSet.has(wellName)) {
                    const newState = mode === 'select' ? true : false;
                    if (well.selected !== newState) {
                        well.selected = newState;
                        changedCount++;
                    }
                }
            }
            
            // Update the plate view to show new well colors and sites
            if (this.plateNavigator) {
                this.plateNavigator.refreshWellColors(this._microscope_config);
            }
            
            // Save the updated configuration
            this.saveCurrentConfig();
        },


        /**
         *
         * @param {AcquisitionConfig} microscope_config
         * @param {Wellplate} plate
         */
        async setPlate(microscope_config, plate) {
            if (!this.plateNavigator) return;
            await this.plateNavigator.loadPlate(microscope_config, plate);
        },

        /** @type {CoreCurrentState|{}} */
        _state: {},
        /** @returns {CoreCurrentState} */
        get state() {
            if (Object.keys(this._state).length == 0) {
                // Return a default state during initialization instead of throwing
                console.warn("no state available")
                
                return {
                    microscope_name:placeholder_microscope_name,
                    is_busy: false,
                    adapter_state: {
                        is_in_loading_position: false,
                        stage_position:{
                            x_pos_mm:0,
                            y_pos_mm:0,
                            z_pos_mm:0,
                        }
                    },

                    latest_imgs:{},
                    is_streaming:false,
                    current_acquisition_id:undefined,
                    last_acquisition_error:null,
                    last_acquisition_error_timestamp:null,
                };
            }
            //@ts-ignore
            return this._state;
        },
        /**
         * @param {CoreCurrentState} newstate
         */
        set state(newstate) {
            /** @ts-ignore */
            Object.assign(this._state, newstate);
        },
        /** @type {{plategroups: WellPlateGroup[],allplates: Wellplate[]}|null}  */
        _plateinfo: null,
        get plateinfo() {
            if (!this._plateinfo) {
                throw new Error(`bug in plateinfo`);
            }
            return this._plateinfo;
        },
        /** @type {AcquisitionConfig|null} */
        _microscope_config: null,
        get microscope_config() {
            if (!this._microscope_config) {
                throw new Error(`bug in microscope_config`);
            }
            return this._microscope_config;
        },
        set microscope_config(newConfig) {
            this._microscope_config = newConfig;
            if (newConfig) {
                this.saveMicroscopeConfigToStorage();
            }
        },

        /** a copy of this is required often, but non-trivial to construct, so the utility is provided here. */
        get microscope_config_copy() {
            return cloneObject(this.microscope_config);
        },

        /**
         * Parse available filters from machine config
         * @returns {Array<{name: string, handle: string, slot: number}>}
         */
        get availableFilters() {
            const filtersConfigItem = this.getMachineConfigItem('filter.wheel.configuration');
            if (!filtersConfigItem || !filtersConfigItem.value) {
                return [];
            }
            
            try {
                if(!(typeof filtersConfigItem.value == "string")){
                    throw new Error(`invalid type of filter config`);
                }
                const filtersData = json5.parse(filtersConfigItem.value);

                const ret=Array.isArray(filtersData) ? filtersData : [];
                return ret;
            } catch (error) {
                console.warn('Failed to parse filters configuration:', error);
                return [];
            }
        },

        /**
         * Check if filter wheel is available
         * @returns {boolean}
         */
        get isFilterWheelAvailable() {
            const filterWheelConfigItem = this.getMachineConfigItem('filter.wheel.available');
            return filterWheelConfigItem?.value === 'yes';
        },

        /**
         * Get filter options for select dropdown (includes None option and grouped filters)
         * @returns {Array<{name: string, handle: string, isGroup?: boolean, groupLabel?: string}>}
         */
        get filterOptions() {
            /** @type {{name:string,handle:string,isGroup:boolean,groupLabel?:string}[]} */
            const options = [{ name: 'Unspecified', handle: '__UI_NONE__', isGroup: false }];
            
            // Add each filter as a group with slot label
            this.availableFilters.forEach(f => {
                options.push({
                    name: f.name,
                    handle: f.handle,
                    isGroup: true,
                    groupLabel: `Slot ${f.slot}`
                });
            });
            
            return options;
        },

/** used to filter the machine config list */
        machineConfigHandleFilter: "",
        
        /**
         * namespace tree for machine config
         * @type {import('./namespace-parser.js').NamespaceNode[]}
         * */
        machineConfigNamespaces: [],
        
        /** whether to use hierarchical view */
        machineConfigUseNamespaces: true,

        /** indicate of connection to server is currently established */
        isConnectedToServer: false,

        // acquisition status websocket for monitoring acquisition progress
        /** @type {WebSocket|null} */
        acquisition_status_ws: null,
        server_url_input: "",
        /**
         *
         * @param {string?} [url=undefined]
         * @returns
         */
        status_reconnect(url) {
            // if new url is same one, and a connection is already [in process of] being established
            // skip reconnect attempt.
            if (url == this.server_url && this.wsManager.isConnected('status') && this.isConnectedToServer) {
                return;
            }

            // if no url has been provided, reconnect to existing url.
            // if one has been provided, update current url and proceed.
            if (url) {
                this.server_url = url;
                // Close old connection when URL changes so we reconnect to the new server
                this.wsManager.closeConnection('status');
            }

            // reconnection is only attempted if connection is not currently established
            this.isConnectedToServer = false;

            // this is an obscure case that sometimes happens in practice.
            // the underlying bug is probably a state synchronization issue,
            // but we can work around this by just trying again later.
            if (this.server_url == null) {
                console.warn(`server_url is invalid. retrying..`);
                setTimeout(() => this.status_reconnect(url), 200);
                return;
            }

            // Create new WebSocket connection using WebSocketManager (with correct protocol)
            // Only create if not already created/connected
            if (!this.wsManager.getConnection('status')) {
                this.wsManager.createConnection('status', '/ws/get_info/current_state', {
                    onMessage: async (ev) => {
                        try {
                            const data = json5.parse(ev.data);
                            await this.updateMicroscopeStatus(data);

                            // if we got this far, the connection to the server is established
                            this.isConnectedToServer = true;
                        } catch (error) {
                            console.error('Error updating microscope status:', error);
                            // Continue polling - don't break the loop
                        }
                    },
                    onOpen: (ws) => {
                        // Send initial message immediately to trigger server response
                        this.wsManager.send('status', "info");
                        // Schedule polling loop
                        requestAnimationFrame(() => this.status_getstate_loop());
                    },
                    onError: (event) => {
                        this.isConnectedToServer = false;
                        // Reconnection is handled by WebSocketManager
                    },
                    onClose: () => {
                        this.isConnectedToServer = false;
                        // Reconnection is handled by WebSocketManager
                    },
                    autoReconnect: true,
                });
            }
        },

        status_getstate_loop() {
            // Only send if connected; WebSocketManager handles reconnection automatically
            if (this.wsManager.isConnected('status')) {
                // send arbitrary message to receive status update
                this.wsManager.send('status', "info");
            }
            // Schedule next poll iteration regardless of connection status
            // The WebSocketManager will reconnect automatically on errors
            requestAnimationFrame(() => this.status_getstate_loop());
        },

        /**
         * 
         * @param {string} acquisitionId 
         */
        startAcquisitionStatusWebSocket(acquisitionId) {
            try {
                // Close existing connection if any
                if (this.wsManager.getConnection('acquisition_status')) {
                    this.wsManager.closeConnection('acquisition_status');
                }

                this.acquisition_status_ws = this.wsManager.createConnection(
                    'acquisition_status',
                    '/ws/acquisition/status',
                    {
                        onMessage: (ev) => {
                            const data = JSON.parse(ev.data);
                            if(typeof data != "object")throw new Error(`event data has wrong type ${typeof data}`);

                            // Store previous status to detect state changes
                            const previousStatus = this.latest_acquisition_status;
                            this.latest_acquisition_status = data;

                            // Handle acquisition status changes
                            if (data && data.acquisition_status) {
                                // Show acquisition errors when status is CRASHED and there's a message
                                if (data.acquisition_status === 'crashed' && data.message) {
                                    console.log(`Detected crashed acquisition with message: ${data.message}`);
                                    if (!previousStatus || previousStatus.acquisition_status !== 'crashed' || previousStatus.message !== data.message) {
                                        this.showError('Acquisition Failed', data.message);
                                    }
                                }
                            }
                        },
                        onError: (ev) => {
                            console.warn('Acquisition status WebSocket error:', ev);
                            // Don't auto-reconnect for acquisition status - let the main status WebSocket handle lifecycle
                        },
                        onOpen: (ws) => {
                            // Start the polling loop for this acquisition
                            this.acquisitionStatusLoop(acquisitionId);
                        },
                        onClose: () => {
                            // WebSocket closed, stop polling
                        },
                        autoReconnect: false,
                    }
                );

            } catch (error) {
                console.warn('Failed to create acquisition status WebSocket:', error);
            }
        },

        /**
         * 
         * @param {string} acquisitionId 
         * @returns 
         */
        acquisitionStatusLoop(acquisitionId) {
            try {
                if (!this.wsManager.isConnected('acquisition_status')) {
                    return; // WebSocket closed, stop polling
                }

                // Send acquisition_id to get status update
                this.wsManager.send('acquisition_status', JSON.stringify({ acquisition_id: acquisitionId }));

                // Schedule next update
                setTimeout(() => this.acquisitionStatusLoop(acquisitionId), 500); // Poll every 500ms
            } catch (error) {
                console.warn('Acquisition status loop error:', error);
            }
        },

        /**
         * Show warning popup when microscope name changes
         * @param {string} previousName - Previous microscope name
         * @param {string} currentName - Current microscope name
         */
        showMicroscopeNameChangeWarning(previousName, currentName) {
            const message = `The microscope has changed from '${previousName}' to '${currentName}'. The cached configuration may not work correctly with the new microscope. Please reload the page to load the correct configuration.`;

            // Show error using existing error display system
            this.showError('Microscope Configuration Mismatch', message);

            // Enable reload button for this specific warning
            this.centralError.showReloadButton = true;

            console.error('Microscope name change detected:', {
                previous: previousName,
                current: currentName,
                recommendation: 'Reload page to load correct configuration'
            });
        },

        /**
         * Reload the page when the reload button is clicked
         */
        reloadPage() {
            location.reload();
        },

        // this is an annoying workaround (in combination with initManual) because
        // alpine does not actually await an async init before mounting the element.
        // which leads to a whole bunch of errors in the console and breaks
        // some functionalities that depend on fields being initialized on mounting.
        async initSelf() {
            // Test server connection and fall back to mock mode if unavailable
            let useMockMode = false;

            try {
                // Quick health check to see if server is available
                const testResponse = await fetch(`${this.server_url}/api/get_features/hardware_capabilities`, {
                    method: 'POST',
                    headers: [['Content-Type', 'application/json']],
                    body: JSON.stringify({}),
                    signal: AbortSignal.timeout(2000), // 2 second timeout
                });

                if (!testResponse.ok) {
                    throw new Error(`Server returned ${testResponse.status}`);
                }

                // Server is available, use real API client
                console.log('[Seafront] Server connection successful');
                this._api = new APIClient(this.server_url, {
                    onError: (title, message) => this.showError(title, message),
                });

                this._wsManager = new WebSocketManager(this.server_url, {
                    reconnectDelay: 200,
                    onError: (message) => console.warn(message),
                    onConnectionStateChange: (name, isConnected) => {
                        if (name === 'status') {
                            this.isConnectedToServer = isConnected;
                        }
                    },
                });

                useMockMode = false;
            } catch (error) {
                // Server unavailable, use mock mode
                console.warn('[Seafront] Server unavailable, using mock mode:', error.message);
                this._api = new MockAPIClient(this.server_url, {
                    onError: (title, message) => this.showError(title, message),
                });
                this._wsManager = new MockWebSocketManager();
                useMockMode = true;
            }

            this._useMockAPI = useMockMode;
            if (useMockMode) {
                this.showMockModeIndicator();
            }

            this._plateinfo = await this.getPlateTypes();
            this._microscope_config = await this.defaultConfig();

            // Get machine config data to extract microscope name
            const machineConfigData = await this.getMachineDefaults();
            this._microscope_config.machine_config = machineConfigData;

            // Load channels and hardware limits from hardware capabilities
            const hardwareCapabilities = await this.getHardwareCapabilities();
            if (hardwareCapabilities.main_camera_imaging_channels) {
                this._microscope_config.channels = hardwareCapabilities.main_camera_imaging_channels;
            } else {
                console.warn("No channels found in hardware capabilities");
            }

            // Update limits with hardware capabilities if available
            if (hardwareCapabilities.hardware_limits) {
                // Replace hardcoded limits with server-provided hardware limits
                this.limits = hardwareCapabilities.hardware_limits;
            } else {
                console.warn("No hardware limits found in capabilities, using defaults");
            }

            // Now that microscope name is available, try to load saved config from localStorage
            const savedData = this.loadMicroscopeConfigFromStorage();
            if (savedData) {
                const savedConfig = savedData.microscope_config;

                // Save server-provided fields that should NEVER be overwritten by cache
                // These represent hardware capabilities and microscope identity
                const serverProvidedMachineConfig = this._microscope_config.machine_config;
                const serverProvidedChannels = this._microscope_config.channels;
                const serverProvidedPlateWells = this._microscope_config.plate_wells;

                // Merge cached user preferences with server-provided defaults
                // Cached config overwrites server defaults, then we restore hardware-specific fields
                const mergedConfig = { ...this._microscope_config, ...savedConfig };

                // ALWAYS restore server-provided fields (never use cached versions)
                // machine_config contains microscope identity and must always match the server
                mergedConfig.machine_config = serverProvidedMachineConfig;

                // ALWAYS restore channels from server (hardware capabilities), but preserve user settings
                // Start with server's channel list (ensures it matches current hardware)
                mergedConfig.channels = serverProvidedChannels;

                // Apply cached channel settings ONLY if channel list matches (same hardware)
                if (savedConfig.channels && savedConfig.channels.length > 0) {
                    // Check if cached channel list matches server's channel list
                    const serverHandles = new Set(serverProvidedChannels.map(ch => ch.handle));
                    const cachedHandles = new Set(savedConfig.channels.map(ch => ch.handle));

                    const channelsMatch =
                        serverProvidedChannels.length === savedConfig.channels.length &&
                        serverProvidedChannels.every(ch => cachedHandles.has(ch.handle)) &&
                        savedConfig.channels.every(ch => serverHandles.has(ch.handle));

                    if (channelsMatch) {
                        // Channel list matches - safe to merge cached settings
                        const cachedChannelSettingsMap = new Map();
                        savedConfig.channels.forEach(channel => {
                            cachedChannelSettingsMap.set(channel.handle, {
                                analog_gain: channel.analog_gain,
                                exposure_time_ms: channel.exposure_time_ms,
                                illum_perc: channel.illum_perc,
                                num_z_planes: channel.num_z_planes,
                                z_offset_um: channel.z_offset_um,
                                enabled: channel.enabled,
                                filter_handle: channel.filter_handle,
                                delta_z_um: channel.delta_z_um
                            });
                        });

                        // Apply cached settings to server's channel list
                        mergedConfig.channels.forEach(channel => {
                            const cachedSettings = cachedChannelSettingsMap.get(channel.handle);
                            if (cachedSettings) {
                                // Preserve user settings while keeping server-provided name/handle
                                Object.assign(channel, cachedSettings);
                            }
                        });
                    }
                    // If channels don't match, we already set mergedConfig.channels to serverProvidedChannels
                    // So cached settings are discarded (hardware changed)
                }

                // Handle plate_wells: server may provide sparse data (only selected wells)
                // Prefer cached complete grid if available, fall back to server's sparse data
                // The expansion to full grid happens later in manualInit()
                const expectedWellCount = mergedConfig.wellplate_type?.Num_wells_x * mergedConfig.wellplate_type?.Num_wells_y || 0;

                if (savedConfig.plate_wells && savedConfig.plate_wells.length >= expectedWellCount) {
                    // Cache has complete grid - use it (preserves all selections)
                    mergedConfig.plate_wells = savedConfig.plate_wells;
                } else if (savedConfig.plate_wells && savedConfig.plate_wells.length > serverProvidedPlateWells.length) {
                    // Cache has more wells than server - use cache and merge server's data
                    mergedConfig.plate_wells = savedConfig.plate_wells;

                    // Update from sparse server data if present
                    const serverWellMap = new Map();
                    serverProvidedPlateWells.forEach(well => {
                        const key = `${well.col},${well.row}`;
                        serverWellMap.set(key, well.selected);
                    });

                    mergedConfig.plate_wells.forEach(well => {
                        const key = `${well.col},${well.row}`;
                        if (serverWellMap.has(key)) {
                            well.selected = serverWellMap.get(key);
                        }
                    });
                } else {
                    // Use server's data and apply cached selections to it
                    // (This handles case where server has complete grid or cache is empty)
                    mergedConfig.plate_wells = serverProvidedPlateWells;

                    if (savedConfig.plate_wells && savedConfig.plate_wells.length > 0) {
                        const cachedWellSelectionsMap = new Map();
                        savedConfig.plate_wells.forEach(well => {
                            const key = `${well.col},${well.row}`;
                            cachedWellSelectionsMap.set(key, well.selected);
                        });

                        mergedConfig.plate_wells.forEach(well => {
                            const key = `${well.col},${well.row}`;
                            if (cachedWellSelectionsMap.has(key)) {
                                well.selected = cachedWellSelectionsMap.get(key);
                            }
                        });
                    }
                }

                // Store grid.mask (site selections) from savedConfig to restore later (after grid initialization)
                if (savedConfig.grid && savedConfig.grid.mask && savedConfig.grid.mask.length > 0) {
                    this._savedSiteSelections = savedConfig.grid.mask;
                }

                this._microscope_config = mergedConfig;

                // Store the configIsStored status to restore later (after all initialization)
                this._savedConfigIsStored = savedData.configIsStored;
            }

            // Initialize namespace tree for machine config
            this.refreshMachineConfigNamespaces();

            // Load microscope-specific interface settings and theme
            try {
                const loadedTheme = this.loadInterfaceSettings();
                this.theme = loadedTheme;
                // Apply the theme immediately
                this.changeTheme();
            } catch (error) {
                console.warn("Could not load microscope-specific settings during init:", error);
                // Continue with default theme
            }

            // Load cached microscopes list for Settings panel
            try {
                this.loadCachedMicroscopes();
            } catch (error) {
                console.warn("Could not load cached microscopes list during init:", error);
            }

            // init data - get initial state from server
            const currentStateJson = await this.api.post('/api/get_info/current_state', {}, {
                context: 'Get initial microscope state',
                showError: false
            });
            await this.updateMicroscopeStatus(currentStateJson);

            // Initialize WebSocket connection for status updates
            // (onopen handler will start the status_getstate_loop)
            this.status_reconnect();

            // Initialize persistent image WebSocket
            this.initImageWebSocket();
        },

        microscopeConfigAsString(){
            return JSON.stringify(this.microscope_config);
        },

        /**
         *
         * @param {HTMLCanvasElement} el
         * @returns {void}
         */
        initChannelView(el) {
            this.view = new ChannelImageView(el, this.cached_channel_image);
        },
        /** @type {ChannelImageView|null} */
        view: null,
        channelViewNumCols: 3,

        /**
         * call this to update the display for a channel
         * @param {HTMLElement} channelElement must be a valid channel display (with class channel-box-image)
         */
        updateChannelCache(channelElement) {
            if (!this.view) return null;

            let channelView;
            try {
                channelView = this.view.ensureSceneForElement(channelElement);
            } catch (error) {
                console.error(error);
                return null;
            }

            const cachedImage = this.cached_channel_image.get(channelView.channelhandle);
            if (!cachedImage) return null;

            this.view.updateTextureData(channelView, cachedImage);
        },

        /** get total number of images acquired with current config */
        get num_images() {
            const num_sites_xy = this.microscope_config.grid.mask.reduce(
                (o, n) => o + (n.selected ? 1 : 0),
                0,
            );
            const num_sites_xyt =
                num_sites_xy * this.microscope_config.grid.num_t;
            const num_wells = this.microscope_config.plate_wells.reduce(
                (o, n) => o + (n.selected ? 1 : 0),
                0,
            );
            const num_channels = this.microscope_config.channels.reduce(
                (o, n) => o + (n.enabled ? 1 * n.num_z_planes : 0),
                0,
            );

            const total_num_images = num_sites_xyt * num_wells * num_channels;

            return total_num_images;
        },

        /**
         * to keep track of interactive well selection with the mouse cursor
         * @type {AcquisitionWellSiteConfigurationSiteSelectionItem?}
         */
        start_selected_well: null,
        start_pos: { x: 0, y: 0 },
        end_pos: { x: 0, y: 0 },
        /**
         * to keep track of interactive well selection with the mouse cursor
         * @type {AcquisitionWellSiteConfigurationSiteSelectionItem?}
         */
        current_hovered_well: null,

        _backupTooltipConfigEnabled: false,
        /**
         * start selection range at current well.
         * @param {MouseEvent} event
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem} well
         */
        setStartSelectedWell(event, well) {
            this.start_selected_well = well;

            // set center of element as start of selection overlay
            const el = event.currentTarget;
            if (!(el instanceof HTMLElement)) return;
            let ebb = el.getBoundingClientRect();
            this.start_pos.x = ebb.left + ebb.width / 2;
            this.start_pos.y = ebb.top + ebb.height / 2;

            // this is only triggered through dom if something was already selected on enter
            // -> manually trigger on same well on mousedown
            this.selectionContinue(event, well);

            // disable tooltips while in selection mode
            this._backupTooltipConfigEnabled = tooltipConfig.enabled;
            tooltipConfig.enabled = false;
        },
        /**
         * flushed selection range. if well is valid, toggles range.
         * if well is null, disables selection mode.
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem?} well
         */
        async endSelectionRange(well) {
            if (!this.start_selected_well) return;

            // flush selection
            await this.toggleWellSelectionRange(this.start_selected_well, well);
            
            // save config after well selection changes
            this.saveCurrentConfig();

            // remove selection overlay
            if (this.overlayelement) {
                this.overlayelement.remove();
                this.overlayelement = null;
            }
            // clear descriptive text
            this.selectionRangeText = "";

            // clear selection elements
            this.start_selected_well = null;
            this.current_hovered_well = null;

            // restore tooltip status
            tooltipConfig.enabled = this._backupTooltipConfigEnabled;
        },
        /** @type {HTMLElement?} */
        overlayelement: null,
        /** descriptive text for the current selection */
        selectionRangeText: "",
        /**
         *
         * @param {MouseEvent} event
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem} well
         */
        selectionContinue(event, well) {
            if (this.start_selected_well) {
                this.current_hovered_well = well;

                // set center of element as end of selection overlay
                const el = event.currentTarget;
                if (!(el instanceof HTMLElement)) return;
                let ebb = el.getBoundingClientRect();
                this.end_pos.x = ebb.left + ebb.width / 2;
                this.end_pos.y = ebb.top + ebb.height / 2;

                if (!this.overlayelement) {
                    this.overlayelement = document.createElement("div");
                    this.overlayelement.style.setProperty(
                        "position",
                        "absolute",
                    );

                    this.overlayelement.style.setProperty(
                        "background-color",
                        "red",
                    );
                    this.overlayelement.style.setProperty("opacity", "0.5");
                    this.overlayelement.style.setProperty("z-index", "1");
                    this.overlayelement.style.setProperty(
                        "pointer-events",
                        "none",
                    );

                    document.body.appendChild(this.overlayelement);
                }

                const top = Math.min(this.start_pos.y, this.end_pos.y);
                const bottom = Math.max(this.start_pos.y, this.end_pos.y);
                const right = Math.max(this.end_pos.x, this.start_pos.x);
                const left = Math.min(this.end_pos.x, this.start_pos.x);

                // start and end position on screen is at center of well.
                // use element size (which is identical for all wells)
                // to extend the overlay to cover all wells in range completely.
                const element_width = ebb.width;
                const element_height = ebb.height;

                // update position (and size)
                this.overlayelement.style.setProperty(
                    "top",
                    `${top - element_height / 2}px`,
                );
                this.overlayelement.style.setProperty(
                    "left",
                    `${left - element_width / 2}px`,
                );
                this.overlayelement.style.setProperty(
                    "width",
                    `${right - left + element_height}px`,
                );
                this.overlayelement.style.setProperty(
                    "height",
                    `${bottom - top + element_width}px`,
                );

                // update descriptive text
                const state_change_text = this.start_selected_well.selected
                    ? "deselect"
                    : "select";

                const start_well_name = this.wellName(this.start_selected_well);
                const end_well_name = this.wellName(this.current_hovered_well);
                if (start_well_name == null || end_well_name == null)
                    throw new Error(`wellname is null in overlay text generation`);

                // in the text, the well that is more to the top left should be first,
                // regardless of manual selection order (hence synthesize corners for
                // text generation).
                const start_well = {
                    row: Math.min(
                        this.start_selected_well.row,
                        this.current_hovered_well.row,
                    ),
                    col: Math.min(
                        this.start_selected_well.col,
                        this.current_hovered_well.col,
                    ),
                    selected: true,
                };
                const end_well = {
                    row: Math.max(
                        this.start_selected_well.row,
                        this.current_hovered_well.row,
                    ),
                    col: Math.max(
                        this.start_selected_well.col,
                        this.current_hovered_well.col,
                    ),
                    selected: true,
                };

                this.selectionRangeText = `will ${state_change_text} ${this.wellName(start_well)} - ${this.wellName(end_well)}`;
            }
        },

        /**
         * set status of all wells in selection to inverse of current status of first selected element
         *
         * if from or to is null, disables selection mode
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem?} from
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem?} to
         */
        async toggleWellSelectionRange(from, to) {
            if (!from || !to) {
                this.start_selected_well = null;
                return;
            }

            const target_status = !from.selected;

            const lower_row = Math.min(from.row, to.row);
            const higher_row = Math.max(from.row, to.row);
            const lower_col = Math.min(from.col, to.col);
            const higher_col = Math.max(from.col, to.col);

            this.microscope_config.plate_wells.forEach((well) => {
                if (well.row < 0) return;
                if (well.col < 0) return;

                if (well.row >= lower_row && well.row <= higher_row) {
                    if (well.col >= lower_col && well.col <= higher_col) {
                        well.selected = target_status;
                    }
                }
            });
            
            // Update the plate view to show new well colors  
            if (this.plateNavigator) {
                this.plateNavigator.refreshWellColors(this.microscope_config);
            }
        },

        /**
         *
         * @param {AcquisitionWellSiteConfigurationSiteSelectionItem} well
         */
        wellInRange(well) {
            if (!this.start_selected_well) return false;
            if (!this.current_hovered_well) return false;

            const row_min = Math.min(
                this.start_selected_well.row,
                this.current_hovered_well.row,
            );
            const row_max = Math.max(
                this.start_selected_well.row,
                this.current_hovered_well.row,
            );
            const col_min = Math.min(
                this.start_selected_well.col,
                this.current_hovered_well.col,
            );
            const col_max = Math.max(
                this.start_selected_well.col,
                this.current_hovered_well.col,
            );

            if (well.col < col_min || well.col > col_max) return false;
            if (well.row < row_min || well.row > row_max) return false;

            return true;
        },

        /** @type {Map<string,CachedChannelImage>} */
        cached_channel_image: new Map(),
        
        /**
         * Track last displayed acquisition error timestamp to avoid showing duplicates
         * @type {string|null}
         */
        _last_displayed_acquisition_error_timestamp: null,

        makeHistogram,
        /**
         *
         * @param {HTMLElement} el
         */
        updateHistogram(el) {
            /** @type {PlotlyTrace[]} */
            const data = [];
            const xvalues = new Uint16Array(257).map((v, i) => i);

            for (const [key, value] of Array.from(
                this.cached_channel_image.entries(),
            ).toSorted((l, r) => {
                return l[0] > r[0] ? 1 : -1;
            })) {
                // key is handle, i.e. key===value.info.channel.handle (which is not terribly useful for displaying)
                const name = value.info.channel.name;

                // skip channels that are not enabled
                if (
                    !(
                        this.microscope_config.channels.find(
                            (c) => c.handle == key,
                        )?.enabled ?? false
                    )
                ) {
                    continue;
                }

                const y = new Float32Array(xvalues.length);

                const rawdata = (() => {
                    switch (value.bit_depth) {
                        case 8:
                            return new Uint8Array(value.data);
                        case 16:
                            return new Uint16Array(value.data).map(
                                (v) => v >> 8,
                            );
                        default:
                            throw new Error(``);
                    }
                })();
                for (const val of rawdata) {
                    y[val]++;
                }
                data.push({
                    name,
                    // @ts-ignore
                    x: xvalues,
                    // @ts-ignore
                    y,
                });
            }
            Plotly.react(el, data, histogramLayout, histogramConfig);
        },

        /**
         * scroll target channel view panel into view
         * @param {string} handle
         */
        channel_makeVisible(handle) {
            const el = document.getElementById(`channel-display-${handle}`);
            // element may not be visible, e.g. because the tab is not currently visible
            if (!el) return;
            el.scrollIntoView({
                behavior: "smooth",
                block: "start",
                inline: "nearest",
            });
        },

        /**
         * rpc to api/acquisition/start
         *
         * internally clones the body.
         * @returns  {Promise<AcquisitionStartResponse>}
         */
        async acquisition_start() {
            try {
                
                await this.machineConfigFlush();

                // Model is always in backend format
                const backend_config = this.microscope_config_copy;

                // mutate copy (to fix some errors we introduce in the interface)
                // 1) remove wells that are unselected or invalid
                backend_config.plate_wells = backend_config.plate_wells.filter(
                    (w) => w.selected && w.col >= 0 && w.row >= 0,
                );

                /** @type {AcquisitionStartRequest} */
                const backend_body = {
                    config_file: backend_config
                };

                return this.api.post('/api/acquisition/start', backend_body, {
                    context: 'Acquisition start',
                    showError: false
                }).catch((error) => {
                    // Show error in central modal
                    this.showError('Acquisition Start Failed', error.message || error.toString());
                    throw error; // Re-throw so calling code still sees the error
                });
            } catch (error) {
                // Handle any errors from machineConfigFlush or config conversion
                //@ts-ignore
                this.showError('Acquisition Start Failed', error.message || error.toString());
                throw error;
            }
        },
        /**
         * rpc to api/acquisition/cancel
         * @returns {Promise<AcquisitionStopResponse>}
         */
        async acquisition_stop() {
            try {
                const acquisitionId = this.state.current_acquisition_id;
                if (!acquisitionId) {
                    throw new Error('No active acquisition to cancel');
                }

                return this.api.post('/api/acquisition/cancel', { acquisition_id: acquisitionId }, {
                    context: 'Acquisition cancel',
                    showError: false
                }).catch((error) => {
                    // Show error in central modal
                    this.showError('Acquisition Cancel Failed', error.message || error.toString());
                    throw error; // Re-throw so calling code still sees the error
                });
            } catch (e) {
                const error = `api/acquisition/cancel failed because ${e}`;
                console.error(error);
                this.showError('Acquisition Cancel Failed', error);
                throw error;
            }
        },
        /** @type {AcquisitionStatusOut?} */
        latest_acquisition_status: null,
        
        get current_acquisition_progress_percent() {
            const images_done =
                this.latest_acquisition_status?.acquisition_progress
                    ?.current_num_images;
            if (images_done == null) return 0;

            const total_images =
                this.latest_acquisition_status?.acquisition_meta_information
                    ?.total_num_images;
            if (total_images == null) return 0;

            return (images_done / total_images) * 100;
        },
        get isAcquisitionRunning() {
            return !!this.state.current_acquisition_id;
        },
        get acquisitionTimeLeftString() {
            const timeLeftSeconds = this.latest_acquisition_status?.acquisition_progress?.estimated_remaining_time_s;
            if (timeLeftSeconds == null || timeLeftSeconds <= 0) {
                return '0h 0m 0s';
            }
            
            const hours = Math.floor(timeLeftSeconds / 3600);
            const minutes = Math.floor((timeLeftSeconds % 3600) / 60);
            const seconds = Math.floor(timeLeftSeconds % 60);
            
            return `${hours}h ${minutes}m ${seconds}s`;
        },
        get acquisitionElapsedTimeString() {
            const elapsedSeconds = this.latest_acquisition_status?.acquisition_progress?.time_since_start_s;
            if (elapsedSeconds == null || elapsedSeconds <= 0) {
                return '0h 0m 0s';
            }

            const hours = Math.floor(elapsedSeconds / 3600);
            const minutes = Math.floor((elapsedSeconds % 3600) / 60);
            const seconds = Math.floor(elapsedSeconds % 60);

            return `${hours}h ${minutes}m ${seconds}s`;
        },
        /**
         *
         * @param {number|null} value
         * @returns
         */
        formatStageCoordXY(value) {
            if (value == null) return '&nbsp;&nbsp;&nbsp;---';
            return value.toFixed(2).padStart(6, '\u00A0'); // Use non-breaking space
        },
        /**
         * 
         * @param {number|null} value 
         * @returns 
         */
        formatStageCoordZ(value) {
            if (value == null) return '&nbsp;&nbsp;&nbsp;---';
            const zInUm = value * 1000;
            return zInUm.toFixed(1).padStart(6, '\u00A0'); // Use non-breaking space
        },
        /**
         * rpc to /api/acquisition/status
         * @param {AcquisitionStatusRequest} body
         * @returns {Promise<AcquisitionStatusResponse>}
         */
        async acquisition_status(body) {
            return this.api.post('/api/acquisition/status', body, {
                context: 'Get acquisition status'
            });
        },

        get Actions() {
            return {
                /**
                 * @param {StoreConfigRequest} body
                 * @returns {Promise<StoreConfigResponse>}
                 */
                storeConfig: async (body) => {
                    return this.api.post('/api/acquisition/config_store', body, {
                        context: 'Save Configuration'
                    });
                },

                /**
                 * @param {LoadConfigRequest} body
                 * @returns {Promise<LoadConfigResponse>}
                 */
                loadConfig: async (body) => {
                    return this.api.post('/api/acquisition/config_fetch', body, {
                        context: 'Load Configuration'
                    });
                },
                /**
                 * rpc to /api/action/move_by
                 * @param {MoveByRequest} body
                 * @returns {Promise<MoveByResult>}
                 */
                moveBy: (body) => {
                    return this.api.post('/api/action/move_by', body, {
                        context: 'Move Stage'
                    });
                },

                /**
                 * rpc to /api/action/move_to
                 * @param {MoveToRequest} body
                 * @returns {Promise<MoveToResult>}
                 */
                moveTo: (body) => {
                    return this.api.post('/api/action/move_to', body, {
                        context: 'Move to position'
                    });
                },

                /**
                 *
                 * @param {MoveToWellRequest} body
                 * @returns {Promise<MoveToWellResponse>}
                 */
                moveToWell: (body) => {
                    return this.api.post('/api/action/move_to_well', body, {
                        context: 'Move to well'
                    });
                },

                /**
                 *
                 * @param {ChannelSnapshotRequest} body
                 * @returns {Promise<ChannelSnapshotResponse>}
                 */
                snapChannel: (body) => {
                    //@ts-ignore
                    return this.machineConfigFlush().then(()=>{
                        // Model is always in backend format
                        const backend_body = body;

                        return this.api.post('/api/action/snap_channel', backend_body, {
                            context: 'Channel Snapshot'
                        });
                    });
                },

                /**
                 * Snap all selected channels with autofocus and z-offsets
                 * @returns {Promise<ChannelSnapSelectionResponse>}
                 */
                snapAllChannels: () => {
                    return this.machineConfigFlush().then(() => {
                        // Create acquisition config with current microscope config
                        const body = {
                            config_file: this.microscope_config
                        };

                        return this.api.post('/api/action/snap_selected_channels', body, {
                            context: 'Snap selected channels'
                        });
                    });
                },

                /**
                 *
                 * @returns {Promise<EnterLoadingPositionResponse>}
                 */
                enterLoadingPosition: () => {
                    return this.api.post('/api/action/enter_loading_position', {}, {
                        context: 'Enter loading position'
                    });
                },
                /**
                 *
                 * @returns {Promise<LeaveLoadingPositionResponse>}
                 */
                leaveLoadingPosition: () => {
                    return this.api.post('/api/action/leave_loading_position', {}, {
                        context: 'Leave loading position'
                    });
                },
                /**
                 * @param {StreamBeginRequest} body
                 * @returns {Promise<StreamingStartedResponse>}
                 */
                streamBegin: async (body) => {
                    await this.machineConfigFlush();

                    // Model is always in backend format
                    const backend_body = body;

                    return this.api.post('/api/action/stream_channel_begin', backend_body, {
                        context: 'Stream channel begin'
                    });
                },
                /**
                 * @param {StreamEndRequest} body
                 * @returns {Promise<StreamEndResponse>}
                 */
                streamEnd: (body) => {
                    // Model is always in backend format
                    const backend_body = body;

                    return this.api.post('/api/action/stream_channel_end', backend_body, {
                        context: 'Stream channel end'
                    });
                },
                /**
                 *
                 * @param {LaserAutofocusCalibrateRequest} body
                 * @returns {Promise<LaserAutofocusCalibrateResponse>}
                 */
                laserAutofocusCalibrate: async (body) => {
                    await this.machineConfigFlush();

                    const response = await this.api.post('/api/action/laser_autofocus_calibrate', body, {
                        context: 'Laser autofocus calibrate'
                    });
                    console.log(response);
                    return response;
                },
                /**
                 *
                 * @param {LaserAutofocusMoveToTargetOffsetRequest} body
                 * @returns {Promise<LaserAutofocusMoveToTargetOffsetResponse>}
                 */
                laserAutofocusMoveToTargetOffset: async (body) => {
                    await this.machineConfigFlush();

                    return this.api.post('/api/action/laser_autofocus_move_to_target_offset', body, {
                        context: 'Laser autofocus move to target offset'
                    });
                },
                /**
                 *
                 * @param {LaserAutofocusMeasureDisplacementRequest} body
                 * @returns {Promise<LaserAutofocusMeasureDisplacementResponse>}
                 */
                laserAutofocusMeasureDisplacement: async (body) => {
                    await this.machineConfigFlush();

                    return this.api.post('/api/action/laser_autofocus_measure_displacement', body, {
                        context: 'Laser autofocus measure displacement'
                    });
                },
                /**
                 *
                 * @param {LaserAutofocusSnapRequest} body
                 * @returns {Promise<LaserAutofocusSnapResponse>}
                 */
                laserAutofocusSnap: async (body) => {
                    await this.machineConfigFlush();

                    return this.api.post('/api/action/snap_reflection_autofocus', body, {
                        context: 'Snap reflection autofocus'
                    });
                },
            };
        },

        /** @type {number} */
        laserAutofocusTargetOffsetUM: 0,
        async button_laserAutofocusMoveToTargetOffset() {
            const res = await this.Actions.laserAutofocusMoveToTargetOffset({
                config_file: this.microscope_config_copy,
                target_offset_um: this.laserAutofocusTargetOffsetUM,
            });
            
            // Automatically measure the current offset after moving to target
            await this.button_laserAutofocusMeasureOffset();
            
            return res;
        },
        /**
         * offset and position where it was measured
         * @type {LaserAutofocusMeasureDisplacementResponse|null}
         */
        laserAutofocusMeasuredOffset: null,
        async button_laserAutofocusMeasureOffset() {
            this.laserAutofocusMeasuredOffset =
                await this.Actions.laserAutofocusMeasureDisplacement({
                    config_file: this.microscope_config_copy,
                });
        },

        /**
         *
         * @param {PlateWellConfig} well
         * @returns {Promise<void>}
         */
        async buttons_wellcontainer_dblclick(well) {
            if (well.col >= 0 && well.row >= 0) {
                await this.Actions.moveToWell({
                    plate_type: this.microscope_config.wellplate_type,
                    well_name: this.wellName(well) ?? "unknownWellName",
                });
            }
        },
        /**
         *
         * @returns {Promise<StreamBeginResponse>}
         */
        async buttons_startStreaming() {
            const target_channel_handle =
                this.actionInput.live_acquisition_channelhandle;
            const target_channel = this.microscope_config.channels.find(
                (c) => c.handle == target_channel_handle,
            );
            if (!target_channel) {
                const error = `could not find a channel with handle '${target_channel_handle}'`;
                console.error(error);
                throw error;
            }

            /** @type {StreamBeginRequest} */
            const body = {
                channel: target_channel,
            };

            try {
                const response = await this.api.post('/api/action/stream_channel_begin', body, {
                    context: 'Start streaming'
                });
                this.isStreaming = true;
                return response;
            } catch (error) {
                console.error("failed to start streaming", error);
                this.isStreaming = false;
                throw error;
            }
        },
        /**
         *
         * @returns {Promise<StreamEndResponse>}
         */
        async buttons_endStreaming() {
            const target_channel_handle =
                this.actionInput.live_acquisition_channelhandle;
            const target_channel = this.microscope_config.channels.find(
                (c) => c.handle == target_channel_handle,
            );
            if (!target_channel) {
                const error = `could not find a channel with handle '${target_channel_handle}'`;
                console.error(error);
                throw error;
            }

            /** @type {StreamEndRequest} */
            const body = {
                channel: target_channel,
            };

            const response = await this.api.post('/api/action/stream_channel_end', body, {
                context: 'End streaming'
            });
            this.isStreaming = false;
            return response;
        },
        /**
         * set this as onchange callback on the select element that controls the streaming channel
         * @param {HTMLSelectElement} element
         */
        callback_setStreamingChannel(element) {
            this.actionInput.live_acquisition_channelhandle = element.value;
        },

        async machineConfigReset() {
            this.microscope_config.machine_config =
                await this.getMachineDefaults();
            this.refreshMachineConfigNamespaces();
        },
        async machineConfigFlush() {
            /** @type {MachineConfigFlushRequest} */
            const body = {
                machine_config: this.microscope_config.machine_config,
            };

            return this.api.post('/api/action/machine_config_flush', body, {
                context: 'Flush machine config'
            });
        },

        // Namespace-related methods
        refreshMachineConfigNamespaces() {
            const machine_config=this.microscope_config?.machine_config;
            if (machine_config) {
                this.machineConfigNamespaces = parseConfigNamespaces(machine_config);
            }
        },

        expandAllNamespaces() {
            expandAllNamespaces(this.machineConfigNamespaces);
        },

        collapseAllNamespaces() {
            collapseAllNamespaces(this.machineConfigNamespaces);
        },

        get filteredMachineConfig() {
            if (this.machineConfigUseNamespaces) {
                return flattenNamespaceTree(this.machineConfigNamespaces, this.machineConfigHandleFilter);
            } else {
                return this.microscope_config?.machine_config?.filter(config => 
                    config.handle.toLowerCase().includes(this.machineConfigHandleFilter.toLowerCase())
                ).sort((a, b) => a.handle.localeCompare(b.handle)) || [];
            }
        },

        /*
        input values that are used by some requests sent to the server, hence
        should be stored here to avoid dom interactions outside alpine
        */
        actionInput: {
            live_acquisition_channelhandle: "",
        },

        /** Track if live streaming is currently active */
        isStreaming: false,

        /** @type {'x'|'y'|'z'} */
        moveByAxis: "z",
        /** move by distance. unit depends on moveByAxis */
        moveByDistance: 1.0,
        /**
         *
         * @param {'+'|'-'} d
         */
        async stageMoveBy(d) {
            let distance_mm = this.moveByDistance;
            // z axis unit is um, xy is mm
            if (this.moveByAxis == "z") {
                distance_mm *= 1e-3;
            }

            switch (d) {
                case "+":
                    {
                        distance_mm *= +1;
                    }
                    break;
                case "-":
                    {
                        distance_mm *= -1;
                    }
                    break;
            }

            await this.Actions.moveBy({ axis: this.moveByAxis, distance_mm });
        },

        get laserAutofocusIsCalibrated() {
            const is_calibrated =
                (this.getMachineConfigItem("laser.autofocus.calibration.is_calibrated")
                    ?.value ?? "no") == "yes";
            return is_calibrated;
        },
        get laserAutofocusReferenceText() {
            const is_calibrated = this.laserAutofocusIsCalibrated;
            const laser_autofocus_calibration_refzmm =
                this.getMachineConfigItem("laser.autofocus.calibration.ref_z_mm");
            if (!is_calibrated || !laser_autofocus_calibration_refzmm) {
                return "(none set)";
            }

            if (laser_autofocus_calibration_refzmm.value_kind != "float") {
                throw new Error(`machine config laser.autofocus.calibration.ref_z_mm has unexpected value kind ${laser_autofocus_calibration_refzmm.value_kind}`);
            }

            const reference_z_mm = laser_autofocus_calibration_refzmm.value;
            return `set at z = ${reference_z_mm.toFixed(3)}`;
        },
        get laserAutofocusReferenceValue() {
            const is_calibrated = this.laserAutofocusIsCalibrated;
            const laser_autofocus_calibration_refzmm =
                this.getMachineConfigItem("laser.autofocus.calibration.ref_z_mm");
            if (!is_calibrated || !laser_autofocus_calibration_refzmm) {
                return 0;
            }

            if (laser_autofocus_calibration_refzmm.value_kind != "float") {
                return 0;
            }

            return laser_autofocus_calibration_refzmm.value;
        },
        /**
         * this calibrates the system and sets the current z as reference
         * -> store for later retrieval
         */
        async buttons_calibrateLaserAutofocusHere() {
            const calibration_data = await this.Actions.laserAutofocusCalibrate(
                {},
            );
            console.log(`calibrated laser autofocus system`, calibration_data);

            const calibration_refzmm = this.getMachineConfigItem(
                "laser.autofocus.calibration.ref_z_mm",
            );
            if (!calibration_refzmm)
                throw new Error(`machine config item laser.autofocus.calibration.ref_z_mm not found during laser autofocus calibration`);
            if (calibration_refzmm.value_kind != "float")
                throw new Error(`machine config item laser.autofocus.calibration.ref_z_mm has unexpected type ${calibration_refzmm.value_kind}`);
            calibration_refzmm.value =
                calibration_data.calibration_data.calibration_position.z_pos_mm;

            const calibration_umpx = this.getMachineConfigItem(
                "laser.autofocus.calibration.um_per_px",
            );
            if (!calibration_umpx)
                throw new Error(`machine config item laser.autofocus.calibration.um_per_px not found during laser autofocus calibration`);
            if (calibration_umpx.value_kind != "float")
                throw new Error(`machine config item laser.autofocus.calibration.um_per_px has unexpected type ${calibration_umpx.value_kind}`);
            calibration_umpx.value =
                calibration_data.calibration_data.um_per_px;

            const calibration_x = this.getMachineConfigItem(
                "laser.autofocus.calibration.x_peak_pos",
            );
            if (!calibration_x)
                throw new Error(`machine config item laser.autofocus.calibration.x_peak_pos not found during laser autofocus calibration`);
            if (calibration_x.value_kind != "float")
                throw new Error(`machine config item laser.autofocus.calibration.x_peak_pos has unexpected type ${calibration_x.value_kind}`);
            calibration_x.value = calibration_data.calibration_data.x_reference;

            const is_calibrated = this.getMachineConfigItem(
                "laser.autofocus.calibration.is_calibrated",
            );
            if (!is_calibrated)
                throw new Error(`machine config item laser.autofocus.calibration.is_calibrated not found during laser autofocus calibration`);
            if (is_calibrated.value_kind != "option")
                throw new Error(`machine config item laser.autofocus.calibration.is_calibrated has unexpected type ${is_calibrated.value_kind}`);
            is_calibrated.value = "yes";
        },

        get laserAutofocusOffsetText() {
            const noresult = "(not measured)";

            const is_calibrated = this.laserAutofocusIsCalibrated;
            if (!is_calibrated) {
                return noresult;
            }
            if (!this.laserAutofocusMeasuredOffset) {
                return noresult;
            }
            return this.laserAutofocusMeasuredOffset.displacement_um;
        },

        laserAutofocusDebug_numz: 7,
        laserAutofocusDebug_totalz_um: 400,
        /** @type {{realz_um:number,measuredz_um:number}[]} */
        laserAutofocusDebug_measurements: [],
        async buttons_laserAutofocusDebugMeasurement() {
            this.laserAutofocusDebug_measurements.length = 0;
            if (!this.laserAutofocusIsCalibrated)
                throw new Error(`in buttons_laserAutofocusDebugMeasurement: laser autofocus is not calibrated`);

            if (this.laserAutofocusDebug_numz < 3)
                throw new Error(`in buttons_laserAutofocusDebugMeasurement: numz (=${this.laserAutofocusDebug_numz}) < 3`);
            const stepDelta_mm =
                (this.laserAutofocusDebug_totalz_um * 1e-3) /
                (this.laserAutofocusDebug_numz - 1);

            const halfz_mm = (this.laserAutofocusDebug_totalz_um * 1e-3) / 2;

            // 1) approach ref z
            /** @ts-ignore @type {number} */
            const refz_mm = this.getMachineConfigItem(
                "laser.autofocus.calibration.ref_z_mm",
            ).value;
            await this.Actions.moveTo({ z_mm: refz_mm });

            // 2) move in steps, measure at each
            for (let i = 0; i < this.laserAutofocusDebug_numz; i++) {
                const current_real_offset_mm = -halfz_mm + i * stepDelta_mm;

                await this.Actions.moveTo({
                    z_mm: refz_mm + current_real_offset_mm,
                });
                try {
                    const res =
                        await this.Actions.laserAutofocusMeasureDisplacement({
                            config_file: this.microscope_config_copy,
                        });
                    this.laserAutofocusDebug_measurements.push({
                        realz_um: current_real_offset_mm * 1e3,
                        measuredz_um: res.displacement_um,
                    });
                } catch (e) { }
            }
            // 3) restore z (by moving to ref)
            await this.Actions.moveTo({ z_mm: refz_mm });

            // 4) flush results
            // nop
        },

        /** @type {HTMLCanvasElement|null} */
        latestLaserAutofocusImageCanvas: null,
        async button_laserAutofocusGetLatestImage() {
            if (!this.latestLaserAutofocusImageCanvas) return;

            const lafSnapRes = await this.Actions.laserAutofocusSnap({
                // @ts-ignore
                exposure_time_ms: this.getMachineConfigItem(
                    "laser.autofocus.exposure_time_ms",
                ).value,
                // @ts-ignore
                analog_gain: this.getMachineConfigItem(
                    "laser.autofocus.camera.analog_gain",
                ).value,
            });

            const img = await this.fetch_image({
                channel: {
                    name: "",
                    handle: "laser_autofocus",
                    analog_gain: 0,
                    exposure_time_ms: 0,
                    illum_perc: 0,
                    num_z_planes: 0,
                    z_offset_um: 0,
                    enabled: true,
                },
                height_px: lafSnapRes.height_px,
                width_px: lafSnapRes.width_px,
                storage_path: "",
                position: { x_pos_mm: 0, y_pos_mm: 0, z_pos_mm: 0 },
                timestamp: 0,
            });
            const imgdata = new ImageData(img.width, img.height);
            const rawimgdata = new Uint8ClampedArray(img.data);
            for (let i = 0; i < img.width * img.height; i++) {
                const px = rawimgdata[i];
                imgdata.data[i * 4 + 0] = px;
                imgdata.data[i * 4 + 1] = px;
                imgdata.data[i * 4 + 2] = px;
                imgdata.data[i * 4 + 3] = 255;
            }

            this.latestLaserAutofocusImageCanvas.width = img.width;
            this.latestLaserAutofocusImageCanvas.height = img.height;
            let ctx = this.latestLaserAutofocusImageCanvas.getContext("2d");
            if (!ctx) return;
            ctx.putImageData(imgdata, 0, 0);
        },

        /**
         *
         * @returns {{data:PlotlyTrace[],layout:PlotlyLayout,config:PlotlyConfig}}
         */
        _getLaserAutofocusDebugMeasurementPlotData() {
            /** @type {{data:PlotlyTrace[],layout:PlotlyLayout,config:PlotlyConfig}} */
            const ret = {
                data: [
                    // measured
                    {
                        x: this.laserAutofocusDebug_measurements.map(
                            (m) => m.realz_um,
                        ),
                        y: this.laserAutofocusDebug_measurements.map(
                            (m) => m.measuredz_um,
                        ),
                        name: "measured",
                        line: {
                            color: "orange",
                        },
                    },
                    // real
                    {
                        x: this.laserAutofocusDebug_measurements.map(
                            (m) => m.realz_um,
                        ),
                        y: this.laserAutofocusDebug_measurements.map(
                            (m) => m.realz_um,
                        ),
                        name: "real",
                        line: {
                            color: "green",
                        },
                    },
                    // error
                    {
                        x: this.laserAutofocusDebug_measurements.map(
                            (m) => m.realz_um,
                        ),
                        y: this.laserAutofocusDebug_measurements.map(
                            (m) => m.measuredz_um - m.realz_um,
                        ),
                        name: "error",
                        line: {
                            color: "red",
                            dash: "dash",
                        },
                    },
                ],
                layout: {
                    autosize: true,
                    showlegend: true,
                    xaxis: {
                        title: { text: "z offset from reference [um]" },
                        range: [
                            // z range, with some margin on either side
                            -10 - this.laserAutofocusDebug_totalz_um / 2,
                            10 + this.laserAutofocusDebug_totalz_um / 2,
                        ],
                    },
                    yaxis: {
                        title: { text: "measured offset [um]" },
                    },
                    margin: {
                        t: 20, // top margin for pan/zoom buttons
                        l: 60, // reduced y axis margin
                        r: 20, // reduced x axis margin
                        b: 40, // bottom margin for x-axis title
                    },
                },
                config: {
                    responsive: true,
                    modeBarButtonsToRemove: [
                        "sendDataToCloud",
                        "zoom2d",
                        "pan2d",
                        "select2d",
                        "lasso2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "resetScale2d",
                    ],
                    showLink: false,
                    displaylogo: false,
                },
            };
            return ret;
        },
        /**
         *
         * @param {HTMLElement} el
         */
        initLaserAutofocusDebugMeasurementDisplay(el) {
            const { data, layout, config } =
                this._getLaserAutofocusDebugMeasurementPlotData();
            Plotly.newPlot(el, data, layout, config);

            new ResizeObserver(function () {
                // @ts-ignore
                Plotly.relayout(el, { autosize: true });
            }).observe(el);
        },
        /**
         *
         * @param {HTMLElement} el
         */
        updateLaserAutofocusDebugMeasurementDisplay(el) {
            const { data, layout, config } =
                this._getLaserAutofocusDebugMeasurementPlotData();
            Plotly.react(el, data, layout, config);
        },

        /**
         *
         * @param {Wellplate} wellplate
         * @returns {PlateWellConfig[]}
         */
        createPlateWells(wellplate) {
            /** @type {PlateWellConfig[]} */
            let new_wells = [];
            // Only create real wells (col >= 0 and row >= 0)
            // Headers are now created dynamically in the HTML templates
            for (let y = 0; y < wellplate.Num_wells_y; y++) {
                for (let x = 0; x < wellplate.Num_wells_x; x++) {
                    /** @type {PlateWellConfig} */
                    let newwell = { col: x, row: y, selected: false };
                    new_wells.push(newwell);
                }
            }
            return new_wells;
        },

        /**
         * Create header wells for display purposes only (not stored in config)
         * @param {Wellplate} wellplate
         * @returns {PlateWellConfig[]}
         */
        createHeaderWells(wellplate) {
            /** @type {PlateWellConfig[]} */
            let header_wells = [];
            
            // Column headers (row = -1)
            for (let x = 0; x < wellplate.Num_wells_x; x++) {
                header_wells.push({ col: x, row: -1, selected: false });
            }
            
            // Row headers (col = -1) 
            for (let y = 0; y < wellplate.Num_wells_y; y++) {
                header_wells.push({ col: -1, row: y, selected: false });
            }
            
            // Corner header (col = -1, row = -1)
            header_wells.push({ col: -1, row: -1, selected: false });
            
            return header_wells;
        },

        /**
         * well name, i.e. location on the plate (e.g. A01).
         *
         * headers (e.g. row headers) have no name.
         * @param {PlateWellConfig} well
         * @returns {string|null}
         */
        wellName(well) {
            const { col: x, row: y } = well;

            const wellisheader = y < 0 || x < 0;
            if (wellisheader) {
                return null;
            }
            return makeWellName(x, y);
        },
        /**
         * text in a well in the navigator.
         *
         * only headers (e.g. row header) have text.
         * @param {PlateWellConfig} well
         * @returns {string|null}
         */
        wellText(well) {
            const { col: x, row: y } = well;

            if (x < 0 && y < 0) {
                // special case for top left corner of well navigator - no text.

                return null;
            } else if (x < 0) {
                // left-most column: row headers

                return makeWellName(1, y).slice(0, 1);
            } else if (y < 0) {
                // top-most row: column headers
                return makeWellName(x, 1).slice(1);
            } else {
                // not a header -> no text

                return null;
            }
        },

        /**
         *
         * @param {Wellplate} plate
         * @returns {number}
         */
        plateNumWells(plate) {
            return plate.Num_wells_x * plate.Num_wells_y;
        },

        /**
         * if newplate_Model_id is a string: update plate navigator and selector to newly selected plate type
         * if newplate_Model_id is not a string: update plate navigator and selector with changed site or well selections
         * @param {string|null|undefined} newplate_Model_id
         * @param {boolean|undefined} force_override
         */
        async updatePlate(newplate_Model_id, force_override) {
            // update plate selector view
            let selectedplate = this.microscope_config.wellplate_type;
            if (typeof newplate_Model_id == "string") {
                const newplate = this.plateinfo.allplates.find(
                    (p) => p.Model_id == newplate_Model_id,
                );
                if (!newplate)
                    throw new Error(`${newplate_Model_id} not found`);
                const oldplate = this.microscope_config.wellplate_type;

                // update refernce to current plate
                selectedplate = newplate;
                this.microscope_config.wellplate_type = newplate;

                // Check if we have well selections that we should preserve
                const currentWells = this.microscope_config.plate_wells || [];
                const hasSelectedWells = currentWells.some(w => w.selected && w.row >= 0 && w.col >= 0);
                
                // generate new wells in the dom
                if (
                    this.plateNumWells(newplate) !=
                    this.plateNumWells(oldplate) ||
                    force_override
                ) {
                    // Don't force override if we have selected wells from localStorage
                    if (force_override && hasSelectedWells) {
                        console.log("🛡️ Skipping well recreation to preserve localStorage selections");
                    } else {
                        console.log("🔧 Creating new well grid");
                        const new_wells = this.createPlateWells(newplate);
                        this.microscope_config.plate_wells = new_wells;
                        
                        // Save config after plate wells are updated
                        this.saveCurrentConfig();
                    }
                }
            }

            /** @type {AcquisitionWellSiteConfigurationSiteSelectionItem[]} */
            const new_masks = [];
            for (let x = 0; x < this.microscope_config.grid.num_x; x++) {
                for (let y = 0; y < this.microscope_config.grid.num_y; y++) {
                    /** @type {AcquisitionWellSiteConfigurationSiteSelectionItem} */
                    const new_mask = { col: x, row: y, selected: true };
                    new_masks.push(new_mask);
                }
            }
            // insert new elements
            this.microscope_config.grid.mask = new_masks;

            // await plate navigator update
            await this.setPlate(this.microscope_config, selectedplate);
            
            // Also explicitly refresh sites to ensure they update with new grid settings
            if (this.plateNavigator) {
                this.plateNavigator.refreshWellColors(this.microscope_config);
            }
        },


        /**
         * Handle site selection based on area selection on the plate
         * Uses AABB intersection testing for precise site selection
         * @param {string[]} wellNames - Array of well names in the selection
         * @param {string} mode - 'select' or 'deselect'
         * @param {{minX: number, maxX: number, minY: number, maxY: number}} selectionBounds - Selection area bounds in mm
         */
        selectSitesByWellArea(wellNames, mode, selectionBounds) {
            if (!this.microscope_config || !this.plateNavigator) {
                console.warn("No plate configuration or navigator available for site selection");
                return;
            }

            // Get grid configuration
            const gridConfig = this.microscope_config.grid;
            const plateConfig = this.microscope_config.wellplate_type;
            
            let modifiedSites = 0;
            
            // Test each selected well
            for (let wellRow = 0; wellRow < plateConfig.Num_wells_y; wellRow++) {
                for (let wellCol = 0; wellCol < plateConfig.Num_wells_x; wellCol++) {
                    // Check if this well is selected
                    const wellSelected = this.microscope_config.plate_wells?.some(w => 
                        w.row === wellRow && w.col === wellCol && w.selected) || false;
                    
                    if (!wellSelected) continue;
                    
                    // Calculate well physical position and dimensions
                    const wellX_mm = plateConfig.Offset_A1_x_mm + wellCol * plateConfig.Well_distance_x_mm;
                    const wellY_mm = plateConfig.Offset_A1_y_mm + wellRow * plateConfig.Well_distance_y_mm;
                    const wellSizeX_mm = plateConfig.Well_size_x_mm;
                    const wellSizeY_mm = plateConfig.Well_size_y_mm;
                    
                    // Test sites in this well using AABB intersection
                    for (let site of gridConfig.mask) {
                        // Calculate site position for intersection testing
                        const fovX = this.plateNavigator.objective.fovx;
                        const fovY = this.plateNavigator.objective.fovy;
                        
                        const sitePos = calculateSitePositionPython(
                            wellX_mm, wellY_mm, wellSizeX_mm, wellSizeY_mm,
                            site.col, site.row,
                            gridConfig.num_x, gridConfig.num_y,
                            gridConfig.delta_x_mm, gridConfig.delta_y_mm
                        );

                        // calculateSitePositionPython returns backend/physical coordinates,
                        // which match the coordinate system of selectionBounds (from mouseToPlateCoordinates)

                        // Calculate site AABB using FOV size
                        const siteMinX = sitePos.x - fovX / 2;
                        const siteMaxX = sitePos.x + fovX / 2;
                        const siteMinY = sitePos.y - fovY / 2;
                        const siteMaxY = sitePos.y + fovY / 2;
                        
                        // AABB intersection test: rectangles intersect if they are NOT separated in any direction
                        const intersects = !(
                            siteMaxX < selectionBounds.minX ||  // site is to the left of selection
                            siteMinX > selectionBounds.maxX ||  // site is to the right of selection
                            siteMaxY < selectionBounds.minY ||  // site is below selection
                            siteMinY > selectionBounds.maxY     // site is above selection
                        );
                        
                        if (intersects) {
                            const newSelected = (mode === 'select');
                            if (site.selected !== newSelected) {
                                site.selected = newSelected;
                                modifiedSites++;
                            }
                        }
                    }
                }
            }
            
            if (modifiedSites > 0) {
                // Refresh the visual display
                this.plateNavigator.refreshWellColors(this.microscope_config);
                
                // Save the updated configuration
                this.saveCurrentConfig();
            }
        },

        // Channel drag-and-drop functionality
        draggedChannelIndex: null,

        /**
         * 
         * @param {DragEvent} event 
         * @param {*} index 
         */
        startChannelDrag(event, index) {
            if(!(event.target instanceof HTMLElement))throw new Error();
            if(!event.dataTransfer)throw new Error();

            this.draggedChannelIndex = index;
            // Find the parent row container and add dragging class to it
            const rowContainer = event.target.closest('.channel-config-row-container');
            if (rowContainer) {
                rowContainer.classList.add('dragging');
            }
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/html', event.target.outerHTML);
        },

        /**
         * 
         * @param {DragEvent} event 
         */
        handleChannelDragOver(event) {
            if(!event.dataTransfer)throw new Error();

            if (this.draggedChannelIndex !== null) {
                event.preventDefault();
                event.dataTransfer.dropEffect = 'move';
            }
        },

        /**
         * 
         * @param {DragEvent} event 
         */
        handleChannelDragEnter(event) {
            if(!(event.target instanceof HTMLElement))throw new Error();
            if(!event.dataTransfer)throw new Error();

            if (this.draggedChannelIndex !== null) {
                event.preventDefault();
                event.target.closest('.channel-config-row-container')?.classList.add('drag-over');
            }
        },

        /**
         * 
         * @param {DragEvent} event 
         */
        handleChannelDragLeave(event) {
            if(!(event.target instanceof HTMLElement))throw new Error();
            if(!(event.currentTarget instanceof HTMLElement))throw new Error();
            if(!(event.relatedTarget instanceof HTMLElement))throw new Error();
            if(!event.dataTransfer)throw new Error();


            // Only remove drag-over class if we're actually leaving the element
            if (!event.currentTarget.contains(event.relatedTarget)) {
                event.target.closest('.channel-config-row-container')?.classList.remove('drag-over');
            }
        },

        /**
         * 
         * @param {DragEvent} event 
         * @param {number} dropIndex 
         */
        handleChannelDrop(event, dropIndex) {
            if(!(event.target instanceof HTMLElement))throw new Error();

            if (this.draggedChannelIndex === null) return;
            
            event.preventDefault();
            event.target.closest('.channel-config-row-container')?.classList.remove('drag-over');
            
            const sourceIndex = this.draggedChannelIndex;
            
            if (sourceIndex !== dropIndex) {
                // Create a new array with the moved channel
                const channels = [...this.microscope_config.channels];
                const [movedChannel] = channels.splice(sourceIndex, 1);
                channels.splice(dropIndex, 0, movedChannel);
                
                // Update the channels array
                this.microscope_config.channels = channels;
                
                // Save the configuration
                this.saveCurrentConfig();
            }
        },

        /**
         * 
         * @param {DragEvent} event 
         */
        endChannelDrag(event) {
            if(!(event.target instanceof HTMLElement))throw new Error();

            // Remove dragging class from the parent row container
            const rowContainer = event.target.closest('.channel-config-row-container');
            if (rowContainer) {
                rowContainer.classList.remove('dragging');
            }
            // Remove drag-over class from all elements
            document.querySelectorAll('.channel-config-row-container.drag-over').forEach(el => {
                el.classList.remove('drag-over');
            });
            this.draggedChannelIndex = null;
        },

        /**
         * Validate filter selection and show/hide error state
         * Called via x-effect to update validation when channel or filter wheel availability changes
         * @param {HTMLSelectElement} selectEl - The filter select element
         * @param {AcquisitionChannelConfig} channel - The channel object
         * @param {boolean} isFilterWheelAvailable - Whether filter wheel is available
         */
        setupFilterValidation(selectEl, channel, isFilterWheelAvailable) {
            if (!(selectEl instanceof HTMLSelectElement)) {
                return;
            }

            let hasError = false;
            const errorMessage = "Filter selection is required when filter wheel is available";

            // Only validate if filter wheel is available and channel is enabled
            if (isFilterWheelAvailable && channel && channel.enabled) {
                if (!channel.filter_handle || channel.filter_handle === '__UI_NONE__') {
                    hasError = true;
                }
            }

            // Update error styling
            if (hasError) {
                selectEl.classList.add("filter-select-error");

                // Remove any existing tooltips for this element
                const existingTooltips = Array.from(document.querySelectorAll('.number-input-tooltip')).filter(
                    /// @ts-ignore
                    tooltip => tooltip._associatedElement === selectEl
                );
                existingTooltips.forEach(tooltip => tooltip.remove());

                // Create error tooltip
                /** @ts-ignore @type {HTMLElement&{_associatedElement:HTMLElement}} */
                const tooltip = document.createElement('div');
                tooltip.className = 'number-input-tooltip';
                tooltip.textContent = errorMessage; // Explicitly set the text content
                tooltip._associatedElement = selectEl;

                // Position tooltip
                const rect = selectEl.getBoundingClientRect();
                document.body.appendChild(tooltip);
                const tooltipRect = tooltip.getBoundingClientRect();

                let left = rect.left;
                let top = rect.top - tooltipRect.height - 8;
                let isAbove = true;

                if (top < 10) {
                    top = rect.bottom + 8;
                    isAbove = false;
                }

                const maxTop = window.innerHeight - tooltipRect.height - 10;
                if (top > maxTop && !isAbove) {
                    top = maxTop;
                }

                const maxLeft = window.innerWidth - tooltipRect.width - 10;
                if (left > maxLeft) {
                    left = maxLeft;
                }
                if (left < 10) {
                    left = 10;
                }

                const elementCenter = rect.left + rect.width / 2;
                let arrowLeft = elementCenter - left;
                const minArrowLeft = 10;
                const maxArrowLeft = tooltipRect.width - 20;
                if (arrowLeft < minArrowLeft) arrowLeft = minArrowLeft;
                if (arrowLeft > maxArrowLeft) arrowLeft = maxArrowLeft;

                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
                tooltip.classList.add(isAbove ? 'tooltip-above' : 'tooltip-below');
                tooltip.style.setProperty('--arrow-left', arrowLeft + 'px');
            } else {
                selectEl.classList.remove("filter-select-error");
                // Remove any existing tooltips for this element
                const existingTooltips = Array.from(document.querySelectorAll('.number-input-tooltip')).filter(
                    /// @ts-ignore
                    tooltip => tooltip._associatedElement === selectEl
                );
                existingTooltips.forEach(tooltip => tooltip.remove());
            }
        },
    }));
});
