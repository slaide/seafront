"use strict";

/**
 * @typedef {Object} APIClientOptions
 * @property {(title: string, message: string) => void} [onError] - Error callback
 */

/**
 * @typedef {Object} ConflictErrorDetail
 * @property {string} message - Error message describing the conflict
 * @property {string[]} busy_reasons - List of reasons why the resource is busy
 */

/**
 * @typedef {Object} ConflictErrorModel
 * @property {ConflictErrorDetail} detail - Conflict error details from server
 */

/**
 * @typedef {Object} WebSocketManagerOptions
 * @property {number} [reconnectDelay=200] - Delay (ms) before reconnecting
 * @property {(message: string) => void} [onError] - Error callback
 * @property {(name: string, isConnected: boolean) => void} [onConnectionStateChange] - Connection state change callback
 */

/**
 * @typedef {Object} WebSocketConnectionOptions
 * @property {(event: MessageEvent) => void} [onMessage] - Message handler
 * @property {(ws: WebSocket) => void} [onOpen] - Open handler
 * @property {() => void} [onClose] - Close handler
 * @property {(event: Event) => void} [onError] - Error handler
 * @property {boolean} [autoReconnect=true] - Whether to auto-reconnect on error/close
 */

/**
 * @typedef {Object} WebSocketConnectionData
 * @property {WebSocket} ws - The WebSocket instance
 * @property {string} endpoint - The WebSocket endpoint path
 * @property {WebSocketConnectionOptions} options - Connection options
 * @property {number} reconnectAttempts - Number of reconnection attempts made
 */

/**
 * @typedef {Object} WebSocketConnectionStatus
 * @property {boolean} connected - Whether the connection is OPEN
 * @property {number|null} readyState - WebSocket readyState (0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED)
 * @property {number} reconnectAttempts - Number of reconnection attempts made
 */

/**
 * Centralized API client for all server communication.
 * Handles REST API calls, error handling, and response validation.
 */
class APIClient {
    /**
     * @param {string} baseURL - Base URL for the API (e.g., window.location.origin)
     * @param {APIClientOptions} [options] - Configuration options
     */
    constructor(baseURL, options = {}) {
        this.baseURL = baseURL;
        this.onError = options.onError || (() => {});
    }

    /**
     * Build full URL from endpoint path
     * @private
     * @param {string} endpoint - API endpoint path (e.g., '/api/acquisition/start')
     * @returns {string} - Full URL (e.g., 'http://localhost:8000/api/acquisition/start')
     */
    buildUrl(endpoint) {
        return `${this.baseURL}${endpoint}`;
    }

    /**
     * Handle API response and check for errors
     * @private
     * @template TResponse - Type of response data
     * @param {Response} response - Fetch Response object
     * @param {string} [context='API request'] - Context for error messages
     * @param {boolean} [showError=true] - Whether to show error UI on failure
     * @returns {Promise<TResponse>} - Parsed response data if successful
     * @throws {Error} - On HTTP errors (500, 409, or other non-2xx responses)
     */
    async handleResponse(response, context = 'API request', showError = true) {
        if (!response.ok) {
            if (response.status === 500) {
                // Handle internal server errors
                const errorBody = await response.json();
                const error = `${context} failed with ${response.statusText} ${response.status} because: ${JSON.stringify(errorBody)}`;
                console.error(error);

                if (showError) {
                    this.onError(context + ' Error', errorBody.detail);
                }
                throw new Error(error);
            } else if (response.status === 409) {
                // Handle conflict errors (e.g., microscope busy)
                /** @type {ConflictErrorModel} */
                const errorBody = await response.json();
                const reasons = errorBody.detail.busy_reasons.map(r => `  • ${r}`).join('\n');
                const userMessage = `${errorBody.detail.message}:\n${reasons}`;

                const error = `${context} failed with ${response.statusText} ${response.status} because: ${userMessage}`;
                console.error(error);

                if (showError) {
                    this.onError(context + ' Error', userMessage);
                }
                throw new Error(error);
            } else {
                const responseBlob = await response.blob();
                const responseText = await responseBlob.text();
                try {
                    JSON.parse(responseText);
                } catch (e) {
                    // Not JSON, ignore
                }
                const error = `${context} failed: ${response.status} ${responseText}`;
                if (showError) {
                    this.onError(context + ' Error', `HTTP ${response.status}: ${responseText}`);
                }
                throw new Error(error);
            }
        }

        return await response.json();
    }

    /**
     * Make a POST request to the API
     * @template TRequest - Type of request body
     * @template TResponse - Type of response data
     * @param {string} endpoint - API endpoint path (e.g., '/api/acquisition/start')
     * @param {TRequest} [body] - Request body object
     * @param {Object} [options] - Additional options
     * @param {string} [options.context='API request'] - Context for error messages
     * @param {boolean} [options.showError=true] - Whether to show error UI on failure
     * @returns {Promise<TResponse>} - Response data from server
     */
    async post(endpoint, body, options = {}) {
        const {
            context = 'API request',
            showError = true,
        } = options;

        const url = this.buildUrl(endpoint);
        const response = await fetch(url, {
            method: 'POST',
            body: JSON.stringify(body),
            headers: [['Content-Type', 'application/json']],
        });

        return this.handleResponse(response, context, showError);
    }

    /**
     * Make a GET request to the API
     * @template TResponse - Type of response data
     * @param {string} endpoint - API endpoint path (e.g., '/api/get_info/current_state')
     * @param {Object} [options] - Additional options
     * @param {string} [options.context='API request'] - Context for error messages
     * @param {boolean} [options.showError=true] - Whether to show error UI on failure
     * @returns {Promise<TResponse>} - Response data from server
     */
    async get(endpoint, options = {}) {
        const {
            context = 'API request',
            showError = true,
        } = options;

        const url = this.buildUrl(endpoint);
        const response = await fetch(url, {
            method: 'GET',
            headers: [['Content-Type', 'application/json']],
        });

        return this.handleResponse(response, context, showError);
    }
}

/**
 * Manages WebSocket connections with automatic reconnection and lifecycle management.
 * Handles multiple named connections (e.g., status, image, acquisition).
 */
class WebSocketManager {
    /**
     * @param {string} baseURL - Base URL for WebSocket connections (e.g., window.location.origin)
     * @param {WebSocketManagerOptions} [options] - Configuration options
     */
    constructor(baseURL, options = {}) {
        this.baseURL = baseURL;
        this.reconnectDelay = options.reconnectDelay || 200;
        this.onError = options.onError || (() => {});
        this.onConnectionStateChange = options.onConnectionStateChange || (() => {});

        /** @type {Map<string, WebSocketConnectionData>} */
        this.connections = new Map();
    }

    /**
     * Build full WebSocket URL from endpoint path
     * @private
     * @param {string} endpoint - WebSocket endpoint path (e.g., '/ws/get_info/current_state')
     * @returns {string} - Full WebSocket URL (e.g., 'ws://localhost:8000/ws/get_info/current_state')
     */
    buildUrl(endpoint) {
        const protocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
        const baseWithoutProtocol = this.baseURL.replace(/^https?:\/\//, '');
        return `${protocol}://${baseWithoutProtocol}${endpoint}`;
    }

    /**
     * Create or retrieve a WebSocket connection
     * @param {string} name - Connection name (e.g., 'status', 'image', 'acquisition')
     * @param {string} endpoint - WebSocket endpoint path (e.g., '/ws/get_info/current_state')
     * @param {WebSocketConnectionOptions} [options] - Configuration options
     * @returns {WebSocket|null} - The WebSocket instance, or null if creation failed
     */
    createConnection(name, endpoint, options = {}) {
        const {
            onMessage = () => {},
            onOpen = () => {},
            onClose = () => {},
            onError = () => {},
            autoReconnect = true,
        } = options;

        // Close existing connection if present
        const existing = this.connections.get(name);
        if (existing) {
            if (existing.ws && existing.ws.readyState !== WebSocket.CLOSED) {
                existing.ws.close();
            }
        }

        try {
            const url = this.buildUrl(endpoint);
            const ws = new WebSocket(url);

            const connectionData = {
                ws,
                endpoint,
                options: {
                    onMessage,
                    onOpen,
                    onClose,
                    onError,
                    autoReconnect,
                },
                reconnectAttempts: 0,
            };

            this.connections.set(name, connectionData);

            ws.onopen = () => {
                connectionData.reconnectAttempts = 0;
                this.onConnectionStateChange(name, true);
                onOpen(ws);
            };

            ws.onmessage = (event) => {
                onMessage(event);
            };

            ws.onerror = (event) => {
                this.onConnectionStateChange(name, false);
                onError(event);

                // Only schedule reconnect if this is still the current connection
                const currentConn = this.connections.get(name);
                if (autoReconnect && currentConn && currentConn.ws === ws) {
                    this.scheduleReconnect(name);
                }
            };

            ws.onclose = () => {
                this.onConnectionStateChange(name, false);
                onClose();

                // Only schedule reconnect if this is still the current connection
                const currentConn = this.connections.get(name);
                if (autoReconnect && currentConn && currentConn.ws === ws) {
                    this.scheduleReconnect(name);
                }
            };

            return ws;
        } catch (error) {
            console.warn(`Failed to create WebSocket '${name}':`, error);
            this.onError(`Failed to create WebSocket: ${JSON.stringify(error)}`);
            return null;
        }
    }

    /**
     * Get an existing connection by name
     * @param {string} name - Connection name
     * @returns {WebSocket|null} - The WebSocket instance, or null if not found
     */
    getConnection(name) {
        const connectionData = this.connections.get(name);
        return connectionData ? connectionData.ws : null;
    }

    /**
     * Check if a connection is open
     * @param {string} name - Connection name
     * @returns {boolean} - True if connection exists and is OPEN, false otherwise
     */
    isConnected(name) {
        const ws = this.getConnection(name);
        if(!ws)return false;
        return ws.readyState === WebSocket.OPEN;
    }

    /**
     * Wait for a connection to be ready (OPEN state)
     * @param {string} name - Connection name
     * @param {number} [timeout=5000] - Timeout in milliseconds
     * @returns {Promise<boolean>} - True if connection opened before timeout, false if timeout
     */
    async waitForConnection(name, timeout = 5000) {
        const startTime = Date.now();
        while (Date.now() - startTime < timeout) {
            if (this.isConnected(name)) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        return false;
    }

    /**
     * Send a message through a connection
     * @param {string} name - Connection name
     * @param {string|ArrayBuffer|Uint8Array} data - Data to send
     * @returns {boolean} - True if sent successfully, false if connection not open
     */
    send(name, data) {
        const ws = this.getConnection(name);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(data);
            return true;
        }
        return false;
    }

    /**
     * Send a JSON message through a connection
     * @param {string} name - Connection name
     * @param {*} data - Data to send (will be JSON stringified)
     * @returns {boolean} - True if sent successfully, false if connection not open
     */
    sendJSON(name, data) {
        return this.send(name, JSON.stringify(data));
    }

    /**
     * Schedule a reconnection attempt
     * @private
     * @param {string} name - Connection name to reconnect
     * @returns {void}
     */
    scheduleReconnect(name) {
        const connectionData = this.connections.get(name);
        if (!connectionData) return;

        connectionData.reconnectAttempts++;

        setTimeout(() => {
            if (this.connections.has(name)) {
                this.createConnection(
                    name,
                    connectionData.endpoint,
                    connectionData.options
                );
            }
        }, this.reconnectDelay);
    }

    /**
     * Close a connection
     * @param {string} name - Connection name to close
     * @param {number} [code=1000] - WebSocket close code (e.g., 1000 for normal closure)
     * @param {string} [reason=''] - Optional close reason
     * @returns {void}
     */
    closeConnection(name, code = 1000, reason = '') {
        const connectionData = this.connections.get(name);
        if (connectionData && connectionData.ws) {
            // Prevent auto-reconnect by temporarily disabling it
            const wasAutoReconnect = connectionData.options.autoReconnect;
            connectionData.options.autoReconnect = false;

            if (connectionData.ws.readyState !== WebSocket.CLOSED) {
                connectionData.ws.close(code, reason);
            }

            // Restore setting
            connectionData.options.autoReconnect = wasAutoReconnect;
        }
        this.connections.delete(name);
    }

    /**
     * Close all connections managed by this manager
     * @returns {void}
     */
    closeAll() {
        for (const name of this.connections.keys()) {
            this.closeConnection(name);
        }
    }
}

export { APIClient, WebSocketManager };
