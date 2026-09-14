// Frontend-side types only. Everything that crosses the API is generated into
// seafront-api.d.ts from the server's own schemas - do not add API shapes here, or they will
// drift from the server the way this file used to.
//
// What legitimately lives here:
//  - types the browser needs that never cross the wire (canvas, cache, scene state)
//  - aliases from the names this frontend has always used to the generated ones
//  - HardwareLimits / HardwareLimitValue, which are more precise here than the server model
//    is: HardwareCapabilitiesResponse.hardware_limits is typed dict[str, dict[str, float|int]]
//    in commands.py, so the generated type cannot express the per-field structure yet.

import * as THREE from "three";

declare global {
    type float = number;
    type int = number;

    /**
     * has html status code 409 (Conflict)
     * returned when microscope is busy
     */
    type MicroscopeBusyError = {
        detail: {
            message: string;
            busy_reasons: string[];
        };
    };

    /**
     * All API requests must include target_device for routing validation
     */
    type TargetDeviceRequest = {
        target_device: string;
    };

    type ChannelInfo = {
        channel: AcquisitionChannelConfig;
        height_px: int;
        width_px: int;
        storage_path: string | undefined;
        position: AdapterPosition;
        timestamp: float;
    };

    type CachedChannelImage = {
        height: int;
        width: int;
        bit_depth: int;
        camera_bit_depth: int;
        data: ArrayBuffer;
        info: ChannelInfo;
    };

    type ChannelImageData = {
        width: int;
        height: int;
        data: ArrayBuffer | Uint16Array | Uint8Array;
        texture: THREE.DataTexture;
        mesh: THREE.Mesh;
    };

    type SceneInfo = {
        range: { zoom: number; offsetx: number; offsety: number };
        channelhandle: string;
        scene: THREE.Scene;
        camera: THREE.OrthographicCamera;
        elem: HTMLElement;
        mesh?: THREE.Mesh;
        img?: ChannelImageData;
    };
}

declare global {
    type Pos2 = {
        x: float;
        y: float;
    };
    type AABB = {
        ax: float;
        ay: float;
        bx: float;
        by: float;
    };
}

declare global {
    type CheckMapSquidRequestFn<T, E extends object> = (
        v: Response,
        context?:string,
        showError?:boolean
    ) => Promise<T>;

    type CachedMicroscopeConfig={
        microscope_config:AcquisitionConfig;
        configIsStored:boolean;
        savedAt:string;
    };

    type CachedInterfaceSettings={
        tooltip: {
            enabled: boolean;
            delayMs: number;
        };
        popupHideDelayMs: number;
        busyLingerMs: number;
        theme: string;
        savedAt: string;
    };

    type MicroscopeCacheData = {
        interface_settings?: CachedInterfaceSettings;
        microscope_config?: CachedMicroscopeConfig;
    };

    type WellPlateGroup = {
        label: string;
        numwells: number;
        plates: Wellplate[];
    };

    type MoveToResult = BasicSuccessResponse;
    type MoveToWellResponse = BasicSuccessResponse;

    type ChannelSnapshotRequest = SnapChannelRequest;
    type ChannelSnapshotResponse = ImageAcquiredResponse;

    type ChannelSnapSelectionResponse = ChannelSnapSelectionResult;

    type StreamBeginRequest = StreamChannelBeginRequest;
    type StreamBeginResponse = StreamingStartedResponse;

    type StreamEndRequest = StreamChannelEndRequest;
    type StreamEndResponse = BasicSuccessResponse;

    type EnterLoadingPositionResponse = BasicSuccessResponse;
    type LeaveLoadingPositionResponse = BasicSuccessResponse;

    type HardwareLimitValue = {
        min: number;
        max: number;  
        step: number;
    };

    type HardwareLimits = {
        imaging_exposure_time_ms: HardwareLimitValue;
        imaging_analog_gain_db: HardwareLimitValue;
        imaging_focus_offset_um: HardwareLimitValue;
        imaging_illum_perc: HardwareLimitValue;
        imaging_illum_perc_fluorescence: HardwareLimitValue;
        imaging_illum_perc_brightfield: HardwareLimitValue;
        imaging_number_z_planes: HardwareLimitValue;
        imaging_delta_z_um: HardwareLimitValue;
    };

    type HardwareCapabilities = HardwareCapabilitiesResponse;

    type MachineConfigItem = ConfigItem;
    type MachineDefaults = MachineConfigItem[];

    type ConfigListInfo = ConfigFileInfo;
    type ConfigListEntry = ConfigListInfo;
    type StoreConfigRequest = ConfigStoreRequest;
    type StoreConfigResponse = BasicSuccessResponse;
    type LoadConfigRequest = ConfigFetchRequest;
    type LoadConfigResponse = ConfigFetchResponse;

    type AcquisitionStopRequest = AcquisitionCancelRequest;
    type AcquisitionStopResponse = BasicSuccessResponse;
    type AcquisitionStopError = InternalErrorModel;

    type AdapterPosition = Position;
    type AcquisitionStatusResponse = AcquisitionStatusOut;
    type AcquisitionStartError = InternalErrorModel;

    type LaserAutofocusCalibrateResponse = LaserAutofocusCalibrationResponse;
    type LaserAutofocusMoveToTargetOffsetRequest = AutofocusApproachTargetRequest;
    type LaserAutofocusMoveToTargetOffsetResponse = AutofocusApproachTargetDisplacementResult;
    type LaserAutofocusMeasureDisplacementRequest = AutofocusMeasureDisplacementRequest;
    type LaserAutofocusMeasureDisplacementResponse = AutofocusMeasureDisplacementResult;

    type LaserAutofocusSnapRequest = AutofocusSnapRequest;
    type LaserAutofocusSnapResponse = AutofocusSnapResult;
}

// this line ensures that the 'declare global' are visible by the LSP in other .js files
export {};
