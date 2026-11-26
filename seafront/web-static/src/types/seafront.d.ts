// Frontend-side types only. Everything that crosses the API is generated into
// seafront-api.d.ts from the server's own schemas - do not add API shapes here, or they will
// drift from the server the way this file used to.
//
// What legitimately lives here:
//  - types the browser needs that never cross the wire (canvas, cache, scene state)
//  - aliases from the names this frontend has always used to the generated ones

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

    type ChannelInfo = ImageStoreInfo;

    type CachedChannelImage = {
        capture?: CapturedImageMetadata;
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
        siteEdgeMarginMm?: Record<string, number>;
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
    /**
     * Validation tooltips (src/filterselect.js, src/init.js) carry a back-reference to the
     * input they annotate, so the right one can be found and removed when that input changes.
     */
    interface Element {
        _associatedElement?: Element;
    }

    /**
     * The part of the Alpine component PlateNavigator talks back to.
     *
     * Declared narrowly on purpose: the navigator needs somewhere to report the cursor and a
     * way to reach the API, not the whole UI object. Before this it was typed `object`, which
     * meant neither field could be checked.
     */
    type PlateNavigatorHost = {
        forbiddenAreas: ForbiddenArea[];
        pruneForbiddenWellSelections: () => void;
        plateCursorPosition: Pos2 | null;
        api: {
            post: (endpoint: string, body?: unknown, options?: ApiRequestOptions) => Promise<any>;
        };
    };

    // --- api client plumbing (src/api-client.js, src/mock-api-client.js) ---------------------
    // Shapes the two clients both use. ConflictErrorModel / LockedErrorModel used to be
    // written out here as well; they are generated from the server's models now.

    type APIClientOptions = {
        /** shown to the user when a request fails */
        onError?: (title: string, message: string) => void;
        /** the microscope this client is talking to, sent as target_device */
        getTargetDevice?: () => string;
        /** this client's device lease token, sent as lock_token, if it holds one */
        getLockToken?: () => string | null;
        /**
         * Take the device lease for this client. Called when the server refuses a hardware
         * request because nobody holds the lease, after which the request is retried once.
         */
        ensureLease?: () => Promise<void>;
    };

    /** per-request overrides for APIClient.post / APIClient.get */
    type ApiRequestOptions = {
        /** named in error messages, so the user knows which request failed */
        context?: string;
        showError?: boolean;
        /** take the lease and retry once when the server says nobody holds it */
        leaseRetry?: boolean;
    };

    type WebSocketManagerOptions = {
        /** delay (ms) before reconnecting */
        reconnectDelay?: number;
        onError?: (message: string) => void;
        onConnectionStateChange?: (name: string, isConnected: boolean) => void;
        getTargetDevice?: () => string;
    };

    type WebSocketConnectionOptions = {
        onMessage?: (event: MessageEvent) => void;
        onOpen?: (ws: AnySocket) => void;
        onClose?: () => void;
        onError?: (event: Event) => void;
        autoReconnect?: boolean;
    };

    type WebSocketConnectionData = {
        ws: AnySocket;
        endpoint: string;
        options: WebSocketConnectionOptions;
        reconnectAttempts: int;
        reconnectScheduled: boolean;
        reconnectTimer: ReturnType<typeof setTimeout> | null;
    };

    type WebSocketConnectionStatus = {
        connected: boolean;
        /** WebSocket readyState (0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED) */
        readyState: int | null;
        reconnectAttempts: int;
    };

    // --- offline mock client (src/mock-api-client.js) ---------------------------------------
    // The mock keeps the state a real microscope would, so its shape is written down rather
    // than left as an untyped blob; that is what lets the mock's replies be checked against
    // the generated response types.

    type MockStoredImage = {
        /** 8-bit at low bit depths, 16-bit otherwise */
        imageData: Uint8Array | Uint16Array;
        width: int;
        height: int;
        bitDepth: int;
        cameraBitDepth: int;
        timestamp: float;
    };

    type MockAcquisitionTask = {
        well: { row: int; col: int };
        wellName: string;
        siteIdx: int;
        channel: AcquisitionChannelConfig;
        zIdx: int;
    };

    type MockAcquisition = {
        id: string;
        config: AcquisitionConfig;
        status: AcquisitionStatusStage;
        startTime: float;
        currentImages: int;
        totalImages: int;
        tasks: MockAcquisitionTask[];
        currentTaskIdx: int;
        lastWellName: string | null;
        interval: ReturnType<typeof setInterval> | null;
    };

    /**
     * What MockWebSocketManager hands back in place of a WebSocket: the part of the interface
     * the app actually uses. Declared so the manager's own methods can be checked against it
     * rather than passing `Object` around.
     */
    type MockSocket = {
        /** WebSocket readyState: 1 (OPEN) until close() sets it to 3 (CLOSED) */
        readyState: int;
        /** set by the image socket, which reads binary frames */
        binaryType: BinaryType;
        send: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => void;
        close: () => void;
    };

    /** either socket kind; the app uses only what both provide */
    type AnySocket = WebSocket | MockSocket;

    type MockState = {
        position: Position;
        targetPosition: Position;
        isMoving: boolean;
        movementStartTime: float | null;
        lastUpdateTime: float | null;
        movementSpeed: float;

        is_busy: boolean;
        is_streaming: boolean;
        is_in_loading_position: boolean;
        /** null when no run is in progress, matching CoreCurrentState */
        current_acquisition_id: string | null;
        latest_imgs: { [channelHandle: string]: ImageStoreInfo };

        imageStore: { [channelHandle: string]: MockStoredImage };

        streamingChannel: AcquisitionChannelConfig | null;
        streamingInterval: ReturnType<typeof setInterval> | null;

        acquisition: MockAcquisition | null;
    };
}

// this line ensures that the 'declare global' are visible by the LSP in other .js files
export {};
