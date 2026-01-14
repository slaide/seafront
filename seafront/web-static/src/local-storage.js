"use strict";

/**
 * Browser localStorage functions for the Alpine microscope_state component.
 * Handles caching of microscope configurations and settings in the browser.
 * Data is stored under the key "seafrontdata" and organized by microscope name.
 *
 * @typedef {import('./init.js').AlpineMicroscopeState} AlpineMicroscopeState
 */

/**
 * Get the complete cache data object for the current microscope from browser localStorage.
 * Data is stored under the key "seafrontdata" and organized by microscope name.
 * @this {AlpineMicroscopeState}
 * @returns {MicroscopeCacheData} The cache data object with interface_settings and microscope_config
 */
export function getMicroscopeCache() {
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
}

/**
 * Set the complete cache data object for the current microscope in browser localStorage.
 * Data is stored under the key "seafrontdata" and organized by microscope name.
 * @this {AlpineMicroscopeState}
 * @param {MicroscopeCacheData} cache The cache data object to store
 */
export function setMicroscopeCache(cache) {
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
}

/**
 * Load all cached microscope names from browser localStorage and populate the list.
 * Used by the Settings panel to show which microscopes have cached configurations.
 * @this {AlpineMicroscopeState}
 */
export function loadCachedMicroscopes() {
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
}

/**
 * Delete a cached microscope configuration from browser localStorage.
 * Removes all cached data (interface settings, config) for the specified microscope.
 * @this {AlpineMicroscopeState}
 * @param {string} microscopeName - The name of the microscope to delete
 */
export function deleteCachedMicroscope(microscopeName) {
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
}

/**
 * Save the current microscope configuration to browser localStorage.
 * This persists user settings (well selections, channel configs, etc.) across page reloads.
 * Data is stored per-microscope under the "seafrontdata" key.
 * @this {AlpineMicroscopeState}
 */
export function saveMicroscopeConfigToStorage() {
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
}

/**
 * Load microscope config from browser localStorage for the current microscope.
 * Returns cached user settings that were previously saved via saveMicroscopeConfigToStorage().
 * @this {AlpineMicroscopeState}
 * @returns {CachedMicroscopeConfig|null} The cached config or null if not found
 */
export function loadMicroscopeConfigFromStorage() {
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
                savedAt: "",
            };
        }
    } catch (error) {
        console.error("Failed to load microscope config from localStorage:", error);
    }
    return null;
}

/**
 * Save current config to browser localStorage (debounced).
 * Debounces by 50ms to batch rapid changes and prevent excessive writes.
 * @this {AlpineMicroscopeState}
 */
export function saveCurrentConfig() {
    // Clear any pending save
    if (this._saveTimeout) {
        clearTimeout(this._saveTimeout);
    }

    // Debounce saves by 50ms to batch rapid changes
    this._saveTimeout = setTimeout(() => {
        this.saveMicroscopeConfigToStorage();
        this._saveTimeout = null;
    }, 50);
}

/**
 * Clear all cached data from browser localStorage for the current microscope.
 * Debug method - removes interface settings and config cache.
 * @this {AlpineMicroscopeState}
 */
export function clearSavedConfig() {
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
}
