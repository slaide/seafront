"use strict";

/**
 * Extracted methods for the Alpine microscope_state component.
 * These functions use `this` which Alpine binds when called as methods.
 *
 * @typedef {import('./init.js').AlpineMicroscopeState} AlpineMicroscopeState
 */

import { calculateSitePositionPython } from "platenavigator";

const placeholder_microscope_name = "<undefined>";

/**
 * Handle busy indicator state with delayed hide.
 * @this {AlpineMicroscopeState}
 * @param {boolean} isBusy - Whether the microscope is currently busy
 */
export function handleBusyIndicator(isBusy) {
    if (isBusy) {
        // Microscope is busy - show indicator immediately
        if (this.busyIndicatorTimeout) {
            clearTimeout(this.busyIndicatorTimeout);
            this.busyIndicatorTimeout = null;
        }
        this.busyIndicatorHiding = false;
        this.showBusyIndicator = true;
    } else {
        // Microscope is not busy - hide after delay
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
}

/**
 * Update microscope status from WebSocket data.
 * Handles state sync, acquisition status, image caching, and UI updates.
 * @this {AlpineMicroscopeState}
 * @param {CoreCurrentState} data - Status data from the server
 */
export async function updateMicroscopeStatus(data) {
    const timestamps = [];

    if (data.latest_imgs) {
        const img_keys = Object.keys(data.latest_imgs);
        for (let channelhandle of img_keys) {
            const channel = data.latest_imgs[channelhandle];
            timestamps.push(channel.timestamp);
        }
    }

    // Check for microscope name changes during reconnection
    if (data.microscope_name && this.state && this.state.microscope_name) {
        const previousMicroscopeName = this.state.microscope_name;
        const currentMicroscopeName = data.microscope_name;

        if (previousMicroscopeName !== currentMicroscopeName && previousMicroscopeName !== placeholder_microscope_name) {
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

            if (!image_cache_outdated) {
                continue;
            }

            try {
                const img = await this.fetch_image(channel_info, 2);
                this.cached_channel_image.set(channel_handle, img);
            } catch (err) {
                // Image fetch failed (e.g., WebSocket error) - will retry on next status update
                console.debug(`Failed to fetch image for ${channel_handle}:`, err);
            }
        }
    }

    // Update objective position if PlateNavigator is available and position data exists
    if (this.plateNavigator && this.state.adapter_state && this.state.adapter_state.stage_position) {
        const position = this.state.adapter_state.stage_position;
        this.plateNavigator.updateObjectivePosition(position.x_pos_mm, position.y_pos_mm);
    }
}

/**
 * Handle site selection based on area selection on the plate.
 * Uses AABB intersection testing for precise site selection.
 * @this {AlpineMicroscopeState}
 * @param {string[]} wellNames - Array of well names in the selection
 * @param {string} mode - 'select' or 'deselect'
 * @param {{minX: number, maxX: number, minY: number, maxY: number}} selectionBounds - Selection area bounds in mm
 */
export function selectSitesByWellArea(wellNames, mode, selectionBounds) {
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
}

