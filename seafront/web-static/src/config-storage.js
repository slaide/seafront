"use strict";

/**
 * Protocol/config storage functions for the Alpine microscope_state component.
 * Handles saving, loading, and listing acquisition configurations.
 *
 * @typedef {import('./init.js').AlpineMicroscopeState} AlpineMicroscopeState
 */

/**
 * Get list of stored configurations from server.
 * @this {AlpineMicroscopeState}
 * @returns {Promise<ConfigListResponse>}
 */
export async function getConfigList() {
    return this.api.post('/api/acquisition/config_list', {}, {
        context: 'Get config list'
    });
}

/**
 * Refresh the protocol list from server.
 * @this {AlpineMicroscopeState}
 */
export async function refreshConfigList() {
    this.protocol_list = (await this.getConfigList()).configs;
}

/**
 * Store current configuration to server.
 * @this {AlpineMicroscopeState}
 */
export async function storeConfig() {
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
}

/**
 * Load a configuration from server.
 * @this {AlpineMicroscopeState}
 * @param {ConfigListEntry} protocol - The protocol to load
 */
export async function loadConfig(protocol) {
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
}
