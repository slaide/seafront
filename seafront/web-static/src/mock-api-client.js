"use strict";

/**
 * Standard positions for the mock microscope
 */
const LOADING_POSITION = { x_pos_mm: 0, y_pos_mm: 0, z_pos_mm: 0 };
const BOOTUP_POSITION = { x_pos_mm: 50, y_pos_mm: 30, z_pos_mm: 0 };

/**
 * Shared mock state for all mock components
 * @type {Object}
 */
const sharedMockState = {
    // Current actual position (animated) - start at bootup position
    position: { x_pos_mm: BOOTUP_POSITION.x_pos_mm, y_pos_mm: BOOTUP_POSITION.y_pos_mm, z_pos_mm: BOOTUP_POSITION.z_pos_mm },
    // Target position for movement interpolation
    targetPosition: { x_pos_mm: BOOTUP_POSITION.x_pos_mm, y_pos_mm: BOOTUP_POSITION.y_pos_mm, z_pos_mm: BOOTUP_POSITION.z_pos_mm },
    // Movement state
    isMoving: false,
    movementStartTime: null,
    lastUpdateTime: null,
    movementSpeed: 20.0, // mm/s - constant speed for all axes

    // Microscope state
    is_busy: false,
    is_streaming: false,
    is_in_loading_position: false,
    current_acquisition_id: undefined,
    latest_imgs: {}, // Metadata only (ChannelInfo)

    // Image storage (raw image data + metadata)
    imageStore: {}, // Map of channel_handle -> { imageData: Uint16Array, width, height, bitDepth, timestamp }

    // Streaming state
    streamingChannel: null,
    streamingInterval: null,

    // Acquisition state
    acquisition: null, // { id, config, status, progress, meta, startTime, interval }
};

/**
 * Start movement animation to a target position
 * @param {number} x_mm - Target X position
 * @param {number} y_mm - Target Y position
 * @param {number} z_mm - Target Z position
 */
function startMovement(x_mm, y_mm, z_mm) {
    sharedMockState.targetPosition.x_pos_mm = x_mm;
    sharedMockState.targetPosition.y_pos_mm = y_mm;
    sharedMockState.targetPosition.z_pos_mm = z_mm;

    const now = performance.now();
    sharedMockState.movementStartTime = now;
    sharedMockState.lastUpdateTime = now;
    sharedMockState.isMoving = true;
}

/**
 * Update animated position based on delta time
 * Called by animation loop in MockWebSocketManager
 */
function updateAnimatedPosition() {
    if (!sharedMockState.isMoving) return;

    const now = performance.now();
    const deltaTime = (now - sharedMockState.lastUpdateTime) / 1000; // seconds since last update
    sharedMockState.lastUpdateTime = now;

    const current = sharedMockState.position;
    const target = sharedMockState.targetPosition;
    const speed = sharedMockState.movementSpeed;

    // Calculate distance to target
    const dx = target.x_pos_mm - current.x_pos_mm;
    const dy = target.y_pos_mm - current.y_pos_mm;
    const dz = target.z_pos_mm - current.z_pos_mm;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (distance < 0.001) {
        // Arrived at target
        current.x_pos_mm = target.x_pos_mm;
        current.y_pos_mm = target.y_pos_mm;
        current.z_pos_mm = target.z_pos_mm;
        sharedMockState.isMoving = false;
        return;
    }

    // Move at constant speed using delta time
    const stepDistance = speed * deltaTime;

    if (stepDistance >= distance) {
        // Reached target this frame
        current.x_pos_mm = target.x_pos_mm;
        current.y_pos_mm = target.y_pos_mm;
        current.z_pos_mm = target.z_pos_mm;
        sharedMockState.isMoving = false;
    } else {
        // Move towards target by stepDistance
        const ratio = stepDistance / distance;
        current.x_pos_mm += dx * ratio;
        current.y_pos_mm += dy * ratio;
        current.z_pos_mm += dz * ratio;
    }
}

/**
 * Check if microscope is busy and throw 409 error if so
 * @throws {Error} 409 Conflict error if busy
 */
function checkBusyState() {
    if (sharedMockState.is_busy || sharedMockState.isMoving) {
        const error = new Error('Microscope is busy');
        error.status = 409;
        error.detail = {
            message: 'Cannot execute command: microscope is busy',
            busy_reasons: sharedMockState.isMoving ? ['Stage is moving'] : ['Microscope is busy'],
        };
        throw error;
    }
}

/**
 * Generate random noise image
 * @param {number} width - Image width in pixels
 * @param {number} height - Image height in pixels
 * @param {number} bitDepth - Bit depth (8 or 16)
 * @returns {Uint8Array|Uint16Array} - Random noise image data
 */
function generateNoiseImage(width, height, bitDepth) {
    const pixelCount = width * height;

    if (bitDepth === 8) {
        const imageData = new Uint8Array(pixelCount);
        for (let i = 0; i < pixelCount; i++) {
            imageData[i] = Math.floor(Math.random() * 256);
        }
        return imageData;
    } else {
        // 16-bit
        const maxValue = Math.pow(2, bitDepth) - 1;
        const imageData = new Uint16Array(pixelCount);
        for (let i = 0; i < pixelCount; i++) {
            imageData[i] = Math.floor(Math.random() * (maxValue + 1));
        }
        return imageData;
    }
}

/**
 * Store image for a channel
 * @param {Object} channel - Channel configuration
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} bitDepth - Bit depth
 */
function storeChannelImage(channel, width = 640, height = 640, bitDepth = 16) {
    const timestamp = Date.now() / 1000; // Unix timestamp in seconds
    const imageData = generateNoiseImage(width, height, bitDepth);

    // Store raw image data
    sharedMockState.imageStore[channel.handle] = {
        imageData: imageData,
        width: width,
        height: height,
        bitDepth: bitDepth,
        cameraBitDepth: bitDepth,
        timestamp: timestamp,
    };

    // Store metadata in latest_imgs (matching ImageStoreInfo structure)
    sharedMockState.latest_imgs[channel.handle] = {
        channel: channel,
        width_px: width,
        height_px: height,
        storage_path: undefined,
        position: {
            well_name: "A1",
            site_x: 0,
            site_y: 0,
            site_z: 0,
            x_offset_mm: 0,
            y_offset_mm: 0,
            z_offset_mm: 0,
            position: { ...sharedMockState.position },
        },
        timestamp: timestamp,
    };
}

/**
 * Calculate total number of images for an acquisition
 * @param {*} config - Acquisition config
 * @returns {number} Total number of images
 */
function calculateTotalImages(config) {
    const selectedWells = config.plate_wells.filter(w => w.selected);
    const enabledChannels = config.channels.filter(ch => ch.enabled);

    // Global grid configuration (not per-well)
    const gridMask = config.grid?.mask || [];
    const numSitesPerWell = gridMask.filter(s => s.selected).length;

    let totalImages = 0;
    for (const well of selectedWells) {
        for (const channel of enabledChannels) {
            const numZPlanes = channel.num_z_planes || 1;
            totalImages += numSitesPerWell * numZPlanes;
        }
    }
    return totalImages;
}

/**
 * Start an acquisition simulation
 * @param {string} acquisitionId - Unique acquisition ID
 * @param {*} config - Acquisition config
 */
function startAcquisitionSimulation(acquisitionId, config) {
    const totalImages = calculateTotalImages(config);
    const startTime = Date.now();

    console.log(`[MockAcquisition] Starting acquisition ${acquisitionId} with ${totalImages} total images`);

    // Calculate timing
    const selectedWells = config.plate_wells.filter(w => w.selected);
    const enabledChannels = config.channels.filter(ch => ch.enabled);
    const gridMask = config.grid?.mask || [];
    const numSites = gridMask.filter(s => s.selected).length;

    // Build list of all imaging tasks (well, site, channel combinations)
    const tasks = [];
    for (const well of selectedWells) {
        // Create well name from row/col (e.g., A1, B2, D6)
        const wellName = String.fromCharCode(65 + well.row) + (well.col + 1);

        for (let siteIdx = 0; siteIdx < numSites; siteIdx++) {
            for (const channel of enabledChannels) {
                const numZPlanes = channel.num_z_planes || 1;
                for (let zIdx = 0; zIdx < numZPlanes; zIdx++) {
                    tasks.push({
                        well: well,
                        wellName: wellName,
                        siteIdx: siteIdx,
                        channel: channel,
                        zIdx: zIdx,
                    });
                }
            }
        }
    }

    console.log(`[MockAcquisition] ${tasks.length} total tasks across ${selectedWells.length} wells, ${numSites} sites/well, ${enabledChannels.length} channels`);

    sharedMockState.acquisition = {
        id: acquisitionId,
        config: config,
        status: 'running',
        startTime: startTime,
        currentImages: 0,
        totalImages: totalImages,
        tasks: tasks,
        currentTaskIdx: 0,
        lastWellName: null,
        interval: null,
    };

    sharedMockState.current_acquisition_id = acquisitionId;
    sharedMockState.is_busy = true;

    // Execute tasks sequentially
    function executeNextTask() {
        const acq = sharedMockState.acquisition;
        if (!acq || acq.status !== 'running') return;

        if (acq.currentTaskIdx >= acq.tasks.length) {
            // All tasks complete
            const elapsed = (Date.now() - acq.startTime) / 1000;
            console.log(`[MockAcquisition] Completed! ${acq.totalImages} images in ${elapsed.toFixed(1)}s`);
            acq.status = 'completed';
            sharedMockState.is_busy = false;

            // Clear acquisition after brief delay to let final status updates be sent
            setTimeout(() => {
                sharedMockState.current_acquisition_id = null;
                sharedMockState.acquisition = null;
            }, 2000);
            return;
        }

        const task = acq.tasks[acq.currentTaskIdx];
        let taskDelay = 0;

        // Calculate site position within the well
        const wellPlateType = acq.config.wellplate_type;
        const grid = acq.config.grid;

        if (wellPlateType && grid) {
            // Get well base position (top-left corner)
            const row = task.well.row;
            const col = task.well.col;
            const wellX = wellPlateType.Offset_A1_x_mm + col * wellPlateType.Well_distance_x_mm;
            const wellY = wellPlateType.Offset_A1_y_mm + row * wellPlateType.Well_distance_y_mm;

            // Calculate well center position
            const wellCenterX = wellX + wellPlateType.Well_size_x_mm / 2;
            const wellCenterY = wellY + wellPlateType.Well_size_y_mm / 2;

            // Get site from mask
            const selectedSites = grid.mask.filter(s => s.selected);
            const site = selectedSites[task.siteIdx];

            if (site) {
                // Calculate site offset from well center based on grid configuration
                // Sites are arranged in a grid, centered in the well
                const centerX = (grid.num_x - 1) / 2;
                const centerY = (grid.num_y - 1) / 2;
                const siteOffsetX = (site.col - centerX) * grid.delta_x_mm;
                const siteOffsetY = (site.row - centerY) * grid.delta_y_mm;

                const siteX = wellCenterX + siteOffsetX;
                const siteY = wellCenterY + siteOffsetY;

                // Calculate movement time if position changed
                const deltaX = siteX - sharedMockState.position.x_pos_mm;
                const deltaY = siteY - sharedMockState.position.y_pos_mm;
                const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
                const moveTime = (distance / sharedMockState.movementSpeed) * 1000; // ms

                if (distance > 0.01) { // Only move if distance is significant
                    taskDelay += moveTime;
                    console.log(`[MockAcquisition] Moving to well ${task.wellName} site ${task.siteIdx + 1}/${selectedSites.length} at (${siteX.toFixed(2)}, ${siteY.toFixed(2)})`);
                    startMovement(siteX, siteY, sharedMockState.position.z_pos_mm);
                }
            }
        }

        // Add exposure time for this image
        taskDelay += task.channel.exposure_time_ms || 50;

        // Add overhead (autofocus, settling, etc.)
        taskDelay += 50;

        // Schedule image capture and next task
        setTimeout(() => {
            if (sharedMockState.acquisition && sharedMockState.acquisition.status === 'running') {
                // Actually capture the image for this channel
                storeChannelImage(task.channel);

                sharedMockState.acquisition.currentImages++;
                sharedMockState.acquisition.currentTaskIdx++;

                // Log progress periodically
                const progress = sharedMockState.acquisition.currentImages;
                if (progress % 10 === 0 || progress === sharedMockState.acquisition.totalImages) {
                    const elapsed = (Date.now() - sharedMockState.acquisition.startTime) / 1000;
                    console.log(`[MockAcquisition] Progress: ${progress}/${sharedMockState.acquisition.totalImages} images (${elapsed.toFixed(1)}s elapsed)`);
                }

                executeNextTask();
            }
        }, taskDelay);
    }

    // Start executing tasks
    executeNextTask();
}

/**
 * Stop the current acquisition
 */
function stopAcquisitionSimulation() {
    if (sharedMockState.acquisition) {
        console.log(`[MockAcquisition] Cancelled at ${sharedMockState.acquisition.currentImages}/${sharedMockState.acquisition.totalImages} images`);
        sharedMockState.acquisition.status = 'cancelled';
        sharedMockState.is_busy = false;

        // Clear after delay to let final status be sent
        setTimeout(() => {
            sharedMockState.current_acquisition_id = null;
            sharedMockState.acquisition = null;
        }, 2000);
    }
}

/**
 * Mock API client for offline/development mode.
 * Simulates server responses when the backend is unavailable.
 */
class MockAPIClient {
    constructor(baseURL, options = {}) {
        this.mockState = sharedMockState;
        this.baseURL = baseURL || '';
        this.onError = options.onError || (() => {});
    }

    /**
     * Build full URL from endpoint path (no-op for mock)
     * @param {string} endpoint
     * @returns {string}
     */
    buildUrl(endpoint) {
        return endpoint;
    }

    /**
     * Get mock response for an endpoint
     * @private
     * @param {string} endpoint - API endpoint path
     * @param {*} body - Request body
     * @returns {Promise<*>} - Mock response data
     */
    async getMockResponse(endpoint, body) {
        // Hardware capabilities
        if (endpoint === '/api/get_features/hardware_capabilities') {
            return {
                main_camera_imaging_channels: [
                    {
                        name: "Brightfield",
                        handle: "bfled",
                        analog_gain: 0.0,
                        exposure_time_ms: 10.0,
                        illum_perc: 50.0,
                        num_z_planes: 1,
                        z_offset_um: 0.0,
                        enabled: true,
                        filter_handle: null,
                        delta_z_um: 3.0,
                    },
                    {
                        name: "GFP (488nm)",
                        handle: "fluo488",
                        analog_gain: 0.0,
                        exposure_time_ms: 50.0,
                        illum_perc: 50.0,
                        num_z_planes: 1,
                        z_offset_um: 0.0,
                        enabled: true,
                        filter_handle: "GFP_filter",
                        delta_z_um: 3.0,
                    },
                    {
                        name: "DAPI (405nm)",
                        handle: "fluo405",
                        analog_gain: 0.0,
                        exposure_time_ms: 100.0,
                        illum_perc: 50.0,
                        num_z_planes: 1,
                        z_offset_um: 0.0,
                        enabled: true,
                        filter_handle: "DAPI_filter",
                        delta_z_um: 3.0,
                    },
                ],
                wellplate_types: [
                    // Revvity PhenoPlate 96-well
                    {
                        Manufacturer: "Revvity",
                        Model_name: "PhenoPlate 96-well",
                        Model_id_manufacturer: "6055302",
                        Model_id: "revvity-96-6055302",
                        Num_wells_x: 12,
                        Num_wells_y: 8,
                        Length_mm: 127.76,
                        Width_mm: 85.48,
                        Well_size_x_mm: 6.4,
                        Well_size_y_mm: 6.4,
                        Well_edge_radius_mm: 3.2, // circular (6.4/2)
                        Offset_A1_x_mm: 11.18, // 14.38 - 6.4/2
                        Offset_A1_y_mm: 8.04, // 11.24 - 6.4/2
                        Well_distance_x_mm: 9.0,
                        Well_distance_y_mm: 9.0,
                        Offset_bottom_mm: 0.328, // 0.118 + 0.210
                    },
                    // Revvity PhenoPlate 384-well
                    {
                        Manufacturer: "Revvity",
                        Model_name: "PhenoPlate 384-well",
                        Model_id_manufacturer: "6057800",
                        Model_id: "revvity-384-6057800",
                        Num_wells_x: 24,
                        Num_wells_y: 16,
                        Length_mm: 127.76,
                        Width_mm: 85.48,
                        Well_size_x_mm: 3.26,
                        Well_size_y_mm: 3.26,
                        Well_edge_radius_mm: 0.2,
                        Offset_A1_x_mm: 10.50, // 12.13 - 3.26/2
                        Offset_A1_y_mm: 7.36, // 8.99 - 3.26/2
                        Well_distance_x_mm: 4.5,
                        Well_distance_y_mm: 4.5,
                        Offset_bottom_mm: 0.328, // 0.118 + 0.210
                    },
                ],
                hardware_limits: {
                    imaging_exposure_time_ms: { min: 0.1, max: 1000.0, step: 0.1 },
                    imaging_analog_gain_db: { min: 0.0, max: 24.0, step: 0.1 },
                    imaging_focus_offset_um: { min: -100.0, max: 100.0, step: 0.1 },
                    imaging_illum_perc: { min: 0.0, max: 100.0, step: 0.1 },
                    imaging_illum_perc_fluorescence: { min: 0.0, max: 100.0, step: 0.1 },
                    imaging_illum_perc_brightfield: { min: 0.0, max: 100.0, step: 0.1 },
                    imaging_number_z_planes: { min: 1, max: 100, step: 1 },
                    imaging_delta_z_um: { min: 0.1, max: 100.0, step: 0.1 },
                },
            };
        }

        // Machine defaults (config items)
        if (endpoint === '/api/get_features/machine_defaults') {
            return [
                {
                    name: "Microscope Name",
                    handle: "microscope_name",
                    value_kind: "text",
                    value: "Mock Microscope (Offline)",
                    frozen: true,
                },
            ];
        }

        // Default acquisition config
        if (endpoint === '/api/acquisition/config_fetch') {
            const capabilities = await this.getMockResponse('/api/get_features/hardware_capabilities');
            return {
                file: {
                    project_name: "mock_project",
                    plate_name: "mock_plate",
                    cell_line: "mock_cells",
                    plate_wells: [],
                    grid: {
                        num_x: 3,
                        delta_x_mm: 1.0,
                        num_y: 3,
                        delta_y_mm: 1.0,
                        num_t: 1,
                        delta_t: { h: 0, m: 0, s: 0 },
                        mask: [],
                    },
                    autofocus_enabled: false,
                    comment: "Mock configuration for offline mode",
                    machine_config: [],
                    wellplate_type: capabilities.wellplate_types[0],
                    timestamp: new Date().toISOString(),
                    channels: capabilities.main_camera_imaging_channels,
                    spec_version: { major: 1, minor: 0, patch: 0 },
                },
            };
        }

        // Current microscope state
        if (endpoint === '/api/get_info/current_state') {
            return {
                adapter_state: {
                    is_in_loading_position: this.mockState.is_in_loading_position,
                    stage_position: this.mockState.position,
                },
                latest_imgs: this.mockState.latest_imgs,
                current_acquisition_id: this.mockState.current_acquisition_id,
                is_streaming: this.mockState.is_streaming,
                is_busy: this.mockState.is_busy,
                microscope_name: "Mock Microscope (Offline)",
                last_acquisition_error: null,
                last_acquisition_error_timestamp: null,
            };
        }

        // === ACQUISITION ===

        if (endpoint === '/api/acquisition/config_list') {
            return { configs: [] };
        }

        if (endpoint === '/api/acquisition/config_store') {
            return {};
        }

        if (endpoint === '/api/acquisition/status') {
            return {
                acquisition_id: this.mockState.current_acquisition_id || 'mock_acq_none',
                acquisition_status: 'completed',
                acquisition_progress: {
                    current_num_images: 0,
                    time_since_start_s: 0,
                    start_time_iso: new Date().toISOString(),
                    current_storage_usage_GB: 0,
                    estimated_remaining_time_s: null,
                    last_image: null,
                },
                acquisition_meta_information: {
                    total_num_images: 0,
                    max_storage_size_images_GB: 0,
                },
                acquisition_config: body?.config_file || {},
                message: 'Mock acquisition complete',
            };
        }

        // === ACTION - Stage Movement ===

        if (endpoint === '/api/action/move_to') {
            checkBusyState(); // Block if microscope is busy
            const x = body?.x_mm !== undefined ? body.x_mm : this.mockState.position.x_pos_mm;
            const y = body?.y_mm !== undefined ? body.y_mm : this.mockState.position.y_pos_mm;
            const z = body?.z_mm !== undefined ? body.z_mm : this.mockState.position.z_pos_mm;
            startMovement(x, y, z);
            return {};
        }

        if (endpoint === '/api/action/move_by') {
            checkBusyState(); // Block if microscope is busy
            const axis = body?.axis;
            const distance = body?.distance_mm || 0;
            const current = this.mockState.position;

            // Calculate new target based on current position
            const x = axis === 'x' ? current.x_pos_mm + distance : current.x_pos_mm;
            const y = axis === 'y' ? current.y_pos_mm + distance : current.y_pos_mm;
            const z = axis === 'z' ? current.z_pos_mm + distance : current.z_pos_mm;

            startMovement(x, y, z);
            return { axis, moved_by_mm: distance };
        }

        if (endpoint === '/api/action/move_to_well') {
            checkBusyState(); // Block if microscope is busy
            // Calculate mock position based on well name
            const wellName = body?.well_name || 'A1';
            const row = wellName.charCodeAt(0) - 65; // A=0, B=1, etc.
            const col = parseInt(wellName.substring(1)) - 1;

            const plateType = body?.plate_type;
            if (plateType) {
                const x = plateType.Offset_A1_x_mm + col * plateType.Well_distance_x_mm;
                const y = plateType.Offset_A1_y_mm + row * plateType.Well_distance_y_mm;
                const z = this.mockState.position.z_pos_mm; // Keep current Z
                startMovement(x, y, z);
            }
            return {};
        }

        if (endpoint === '/api/action/machine_config_flush') {
            // Machine config flush - just acknowledge in mock mode
            return {};
        }

        // === ACTION - Streaming ===

        if (endpoint === '/api/action/stream_channel_begin') {
            checkBusyState(); // Block if microscope is busy
            const channel = body?.channel;
            if (channel) {
                // Stop any existing streaming
                if (this.mockState.streamingInterval) {
                    clearInterval(this.mockState.streamingInterval);
                }

                // Start streaming
                this.mockState.is_streaming = true;
                this.mockState.streamingChannel = channel;

                // Generate images at the rate determined by exposure time
                const exposureTimeMs = channel.exposure_time_ms || 33;
                this.mockState.streamingInterval = setInterval(() => {
                    if (this.mockState.is_streaming) {
                        storeChannelImage(channel);
                    }
                }, exposureTimeMs);
            }
            return { channel };
        }

        if (endpoint === '/api/action/stream_channel_end') {
            // Stop streaming
            if (this.mockState.streamingInterval) {
                clearInterval(this.mockState.streamingInterval);
                this.mockState.streamingInterval = null;
            }
            this.mockState.is_streaming = false;
            this.mockState.streamingChannel = null;
            return {};
        }

        // === ACTION - Snapshots ===

        if (endpoint === '/api/action/snap_channel') {
            checkBusyState(); // Block if microscope is busy
            const channel = body?.channel;
            if (channel) {
                // Simulate exposure time
                this.mockState.is_busy = true;
                const exposureTimeMs = channel.exposure_time_ms || 0;
                await new Promise(resolve => setTimeout(resolve, exposureTimeMs));
                storeChannelImage(channel);
                this.mockState.is_busy = false;
            }
            return {};
        }

        if (endpoint === '/api/action/snap_selected_channels') {
            checkBusyState(); // Block if microscope is busy

            // Extract enabled channels from config_file
            const config_file = body?.config_file;
            if (config_file?.channels) {
                const enabledChannels = config_file.channels.filter(ch => ch.enabled);

                // Simulate exposure time for each channel sequentially
                this.mockState.is_busy = true;
                for (const channel of enabledChannels) {
                    const exposureTimeMs = channel.exposure_time_ms || 0;
                    await new Promise(resolve => setTimeout(resolve, exposureTimeMs));
                    storeChannelImage(channel);
                }
                this.mockState.is_busy = false;

                return { channel_handles: enabledChannels.map(ch => ch.handle) };
            }

            return { channel_handles: [] };
        }

        // === ACTION - Loading Position ===

        if (endpoint === '/api/action/enter_loading_position') {
            checkBusyState(); // Block if microscope is busy
            this.mockState.is_in_loading_position = true;
            startMovement(LOADING_POSITION.x_pos_mm, LOADING_POSITION.y_pos_mm, LOADING_POSITION.z_pos_mm);
            return {};
        }

        if (endpoint === '/api/action/leave_loading_position') {
            checkBusyState(); // Block if microscope is busy
            this.mockState.is_in_loading_position = false;
            startMovement(BOOTUP_POSITION.x_pos_mm, BOOTUP_POSITION.y_pos_mm, BOOTUP_POSITION.z_pos_mm);
            return {};
        }

        // === ACTION - Laser Autofocus ===

        if (endpoint === '/api/action/laser_autofocus_calibrate') {
            return {
                calibration_data: {
                    um_per_px: 0.5,
                    x_reference: 320.0,
                    calibration_position: { ...this.mockState.position },
                },
            };
        }

        if (endpoint === '/api/action/laser_autofocus_move_to_target_offset') {
            return {
                num_compensating_moves: 1,
                uncompensated_offset_mm: 0.001,
                reached_threshold: true,
            };
        }

        if (endpoint === '/api/action/laser_autofocus_measure_displacement') {
            return {
                displacement_um: 0.5,
            };
        }

        if (endpoint === '/api/action/snap_reflection_autofocus') {
            return {
                width_px: 640,
                height_px: 480,
            };
        }

        // === ACTION - Configuration ===

        if (endpoint === '/api/action/machine_config_flush') {
            return {};
        }

        // === ACQUISITION ===

        if (endpoint === '/api/acquisition/start') {
            checkBusyState(); // Block if microscope is busy

            const config = body?.config_file;
            if (!config) {
                throw new Error('No config_file provided');
            }

            // Validate acquisition config
            const selectedWells = (config.plate_wells || []).filter(w => w.selected);
            const enabledChannels = (config.channels || []).filter(ch => ch.enabled);

            // Check for wells
            if (selectedWells.length === 0) {
                throw new Error('No wells selected for acquisition');
            }

            // Check for channels
            if (enabledChannels.length === 0) {
                throw new Error('No channels enabled for acquisition');
            }

            // Check that at least one site is selected in the global grid
            const gridMask = config.grid?.mask || [];
            const selectedSites = gridMask.filter(s => s.selected).length;

            if (selectedSites === 0 && gridMask.length > 0) {
                throw new Error('No sites selected in grid. Enable sites in the imaging grid.');
            }

            // Generate unique acquisition ID
            const acquisitionId = `mock_acq_${Date.now()}`;

            // Start acquisition simulation
            startAcquisitionSimulation(acquisitionId, config);

            return { acquisition_id: acquisitionId };
        }

        if (endpoint === '/api/acquisition/cancel') {
            const acquisitionId = body?.acquisition_id;
            if (!acquisitionId || !sharedMockState.acquisition || sharedMockState.acquisition.id !== acquisitionId) {
                throw new Error('No active acquisition to cancel');
            }

            stopAcquisitionSimulation();
            return {};
        }

        if (endpoint === '/api/acquisition/status') {
            const acquisitionId = body?.acquisition_id;
            if (!sharedMockState.acquisition || sharedMockState.acquisition.id !== acquisitionId) {
                throw new Error('Acquisition not found');
            }

            const acq = sharedMockState.acquisition;
            const elapsed = (Date.now() - acq.startTime) / 1000;
            const remainingImages = acq.totalImages - acq.currentImages;

            // Estimate remaining time based on current rate
            let estimatedRemaining = null;
            if (acq.status === 'running' && acq.currentImages > 0) {
                const avgTimePerImage = elapsed / acq.currentImages;
                estimatedRemaining = remainingImages * avgTimePerImage;
            }

            return {
                acquisition_id: acq.id,
                acquisition_status: acq.status,
                acquisition_progress: {
                    current_num_images: acq.currentImages,
                    time_since_start_s: elapsed,
                    start_time_iso: new Date(acq.startTime).toISOString(),
                    current_storage_usage_GB: (acq.currentImages * 0.001), // Mock: 1MB per image
                    estimated_remaining_time_s: estimatedRemaining,
                    last_image: null,
                },
                acquisition_meta_information: {
                    total_num_images: acq.totalImages,
                    max_storage_size_images_GB: (acq.totalImages * 0.001),
                },
                acquisition_config: acq.config,
                message: '',
            };
        }

        if (endpoint === '/api/acquisition/config_store') {
            // Store configuration (mock: just acknowledge)
            return { success: true };
        }

        if (endpoint === '/api/acquisition/config_fetch') {
            // Load configuration (mock: return a minimal config)
            return {
                file: {
                    channels: [],
                    plate_wells: [],
                    grid: { mask: [] },
                    wellplate_type: null,
                    machine_config: [],
                }
            };
        }

        // === ACTION - Dynamic machine config actions ===

        if (endpoint.startsWith('/api/action/')) {
            // Handle dynamic machine config action endpoints
            return {};
        }

        // Default: return empty success response
        console.log(`[MockAPI] Unhandled endpoint: ${endpoint}`);
        return {};
    }

    /**
     * Make a POST request (simulated)
     * @param {string} endpoint
     * @param {*} body
     * @param {Object} [options]
     * @returns {Promise<*>}
     */
    async post(endpoint, body, options = {}) {
        const {
            context = 'API request',
            showError = true,
        } = options;

        try {
            // Instant response (<1ms) - no artificial delay
            return await this.getMockResponse(endpoint, body);
        } catch (error) {
            // Handle 409 Conflict errors (microscope busy)
            if (error.status === 409 && showError) {
                if (error.detail && error.detail.busy_reasons) {
                    const reasons = error.detail.busy_reasons.map(r => `  • ${r}`).join('\n');
                    const userMessage = `${error.detail.message}:\n${reasons}`;
                    this.onError(context + ' Error', userMessage);
                } else {
                    this.onError(context + ' Error', error.message || 'Microscope is busy');
                }
            } else if (showError) {
                this.onError(context + ' Error', error.message || 'Unknown error');
            }
            throw error;
        }
    }

    /**
     * Make a GET request (simulated)
     * @param {string} endpoint
     * @param {Object} [options]
     * @returns {Promise<*>}
     */
    async get(endpoint, options = {}) {
        const {
            context = 'API request',
            showError = true,
        } = options;

        try {
            // Instant response (<1ms) - no artificial delay
            return await this.getMockResponse(endpoint, null);
        } catch (error) {
            if (error.status === 409 && showError) {
                if (error.detail && error.detail.busy_reasons) {
                    const reasons = error.detail.busy_reasons.map(r => `  • ${r}`).join('\n');
                    const userMessage = `${error.detail.message}:\n${reasons}`;
                    this.onError(context + ' Error', userMessage);
                } else {
                    this.onError(context + ' Error', error.message || 'Microscope is busy');
                }
            } else if (showError) {
                this.onError(context + ' Error', error.message);
            }
            throw error;
        }
    }
}

/**
 * Mock WebSocket manager for offline mode.
 * Simulates WebSocket connections with periodic updates.
 */
class MockWebSocketManager {
    constructor() {
        this.connections = new Map();
        this.mockState = sharedMockState;
    }

    /**
     * Build WebSocket URL (no-op for mock)
     * @param {string} endpoint
     * @returns {string}
     */
    buildUrl(endpoint) {
        return endpoint;
    }

    /**
     * Create a mock WebSocket connection
     * @param {string} name
     * @param {string} endpoint
     * @param {Object} options
     * @returns {Object} - Mock WebSocket-like object
     */
    createConnection(name, endpoint, options = {}) {
        const {
            onMessage = () => {},
            onOpen = () => {},
            onClose = () => {},
            onError = () => {},
        } = options;

        // Protocol state for image WebSocket
        const protocolState = {
            waitingFor: 'channel_handle', // or 'downsample_factor'
            currentChannel: null,
        };

        // Create mock WebSocket object
        const mockWs = {
            readyState: 1, // OPEN
            send: (data) => {
                // Handle image WebSocket protocol
                if (endpoint === '/ws/get_info/acquired_image') {
                    if (protocolState.waitingFor === 'channel_handle') {
                        const channelHandle = data;
                        protocolState.currentChannel = channelHandle;

                        // Check if we have an image for this channel
                        const imageEntry = this.mockState.imageStore[channelHandle];

                        if (!imageEntry) {
                            // No image available, send empty JSON
                            onMessage({ data: JSON.stringify({}) });
                            protocolState.waitingFor = 'channel_handle';
                        } else {
                            // Send metadata
                            const metadata = {
                                width: imageEntry.width,
                                height: imageEntry.height,
                                camera_bit_depth: imageEntry.cameraBitDepth,
                                bit_depth: imageEntry.bitDepth,
                            };
                            onMessage({ data: JSON.stringify(metadata) });
                            protocolState.waitingFor = 'downsample_factor';
                        }
                    } else if (protocolState.waitingFor === 'downsample_factor') {
                        const downsampleFactor = parseInt(data, 10) || 1;
                        const channelHandle = protocolState.currentChannel;
                        const imageEntry = this.mockState.imageStore[channelHandle];

                        if (imageEntry) {
                            // Downsample the image
                            const origWidth = imageEntry.width;
                            const origHeight = imageEntry.height;
                            const newWidth = Math.floor(origWidth / downsampleFactor);
                            const newHeight = Math.floor(origHeight / downsampleFactor);

                            const origData = imageEntry.imageData;
                            const bytesPerPixel = imageEntry.bitDepth === 8 ? 1 : 2;
                            const downsampledData = new (imageEntry.bitDepth === 8 ? Uint8Array : Uint16Array)(newWidth * newHeight);

                            // Simple downsampling: take every Nth pixel
                            for (let y = 0; y < newHeight; y++) {
                                for (let x = 0; x < newWidth; x++) {
                                    const origX = x * downsampleFactor;
                                    const origY = y * downsampleFactor;
                                    const origIndex = origY * origWidth + origX;
                                    const newIndex = y * newWidth + x;
                                    downsampledData[newIndex] = origData[origIndex];
                                }
                            }

                            // Send binary data
                            onMessage({ data: downsampledData.buffer });
                        }

                        protocolState.waitingFor = 'channel_handle';
                    }
                }
            },
            close: () => {
                mockWs.readyState = 3; // CLOSED
                this.stopPeriodicUpdates(name);
            },
        };

        this.connections.set(name, {
            ws: mockWs,
            endpoint,
            options,
            protocolState,
        });

        // Call onOpen immediately
        setTimeout(() => onOpen(mockWs), 10);

        // Start periodic updates for status connection only
        if (endpoint === '/ws/get_info/current_state') {
            this.startStatusUpdates(name, onMessage);
        }

        if (endpoint === '/ws/acquisition/status') {
            this.startAcquisitionStatusUpdates(name, onMessage);
        }

        // Image WebSocket is request-response based, no periodic updates needed

        return mockWs;
    }

    /**
     * Start sending periodic status updates
     * @private
     * @param {string} name
     * @param {Function} onMessage
     */
    startStatusUpdates(name, onMessage) {
        // Update at 16ms intervals (60 FPS) for smooth movement animation
        const interval = setInterval(() => {
            const conn = this.connections.get(name);
            if (!conn) {
                clearInterval(interval);
                return;
            }

            // Update animated position for smooth movement
            updateAnimatedPosition();

            const statusUpdate = {
                adapter_state: {
                    is_in_loading_position: this.mockState.is_in_loading_position,
                    stage_position: {
                        x_pos_mm: this.mockState.position.x_pos_mm,
                        y_pos_mm: this.mockState.position.y_pos_mm,
                        z_pos_mm: this.mockState.position.z_pos_mm,
                    },
                },
                latest_imgs: this.mockState.latest_imgs,
                current_acquisition_id: this.mockState.current_acquisition_id,
                is_streaming: this.mockState.is_streaming,
                is_busy: this.mockState.is_busy || this.mockState.isMoving,
                microscope_name: "Mock Microscope (Offline)",
                last_acquisition_error: null,
                last_acquisition_error_timestamp: null,
            };

            const statusJson = JSON.stringify(statusUpdate);

            onMessage({
                data: statusJson,
            });
        }, 16);

        // Store interval for cleanup
        this.connections.get(name).interval = interval;
    }

    /**
     * Start sending periodic acquisition status updates
     * @private
     * @param {string} name
     * @param {Function} onMessage
     */
    startAcquisitionStatusUpdates(name, onMessage) {
        let stopScheduled = false;

        // Update at 200ms intervals for acquisition progress
        const interval = setInterval(() => {
            const conn = this.connections.get(name);
            if (!conn || !this.mockState.acquisition) {
                clearInterval(interval);
                return;
            }

            const acq = this.mockState.acquisition;
            const elapsed = (Date.now() - acq.startTime) / 1000;
            const remainingImages = acq.totalImages - acq.currentImages;

            // Estimate remaining time based on current rate
            let estimatedRemaining = null;
            if (acq.status === 'running' && acq.currentImages > 0) {
                const avgTimePerImage = elapsed / acq.currentImages;
                estimatedRemaining = remainingImages * avgTimePerImage;
            }

            const statusUpdate = {
                acquisition_id: acq.id,
                acquisition_status: acq.status,
                acquisition_progress: {
                    current_num_images: acq.currentImages,
                    time_since_start_s: elapsed,
                    start_time_iso: new Date(acq.startTime).toISOString(),
                    current_storage_usage_GB: (acq.currentImages * 0.001), // Mock: 1MB per image
                    estimated_remaining_time_s: estimatedRemaining,
                    last_image: null,
                },
                acquisition_meta_information: {
                    total_num_images: acq.totalImages,
                    max_storage_size_images_GB: (acq.totalImages * 0.001),
                },
                acquisition_config: acq.config,
                message: '',
            };

            onMessage({
                data: JSON.stringify(statusUpdate),
            });

            // Keep sending updates for 2.5 seconds after completion/cancellation
            // Schedule stop only once when status first changes
            if (acq.status !== 'running' && !stopScheduled) {
                stopScheduled = true;
                setTimeout(() => {
                    clearInterval(interval);
                    console.log('[MockAcquisition] Stopped sending acquisition status updates');
                }, 2500);
            }
        }, 200);

        // Store interval for cleanup
        this.connections.get(name).interval = interval;
    }

    /**
     * Stop periodic updates for a connection
     * @private
     * @param {string} name
     */
    stopPeriodicUpdates(name) {
        const conn = this.connections.get(name);
        if (conn?.interval) {
            clearInterval(conn.interval);
        }
    }

    /**
     * Get an existing connection by name
     * @param {string} name
     * @returns {Object|null}
     */
    getConnection(name) {
        const conn = this.connections.get(name);
        return conn ? conn.ws : null;
    }

    /**
     * Check if a connection is open
     * @param {string} name
     * @returns {boolean}
     */
    isConnected(name) {
        const ws = this.getConnection(name);
        return ws ? ws.readyState === 1 : false;
    }

    /**
     * Wait for a connection to be ready (always resolves immediately for mock)
     * @param {string} name
     * @param {number} [timeout=5000]
     * @returns {Promise<boolean>}
     */
    async waitForConnection(name, timeout = 5000) {
        return this.isConnected(name);
    }

    /**
     * Send a message through a connection
     * @param {string} name
     * @param {*} data
     * @returns {boolean}
     */
    send(name, data) {
        const ws = this.getConnection(name);
        if (ws && ws.readyState === 1) {
            ws.send(data);
            return true;
        }
        return false;
    }

    /**
     * Send a JSON message through a connection
     * @param {string} name
     * @param {*} data
     * @returns {boolean}
     */
    sendJSON(name, data) {
        return this.send(name, JSON.stringify(data));
    }

    /**
     * Close a connection
     * @param {string} name
     */
    closeConnection(name) {
        const conn = this.connections.get(name);
        if (conn) {
            this.stopPeriodicUpdates(name);
            if (conn.ws) {
                conn.ws.close();
            }
            this.connections.delete(name);
        }
    }

    /**
     * Close all connections
     */
    closeAll() {
        for (const name of this.connections.keys()) {
            this.closeConnection(name);
        }
    }
}

export { MockAPIClient, MockWebSocketManager };
