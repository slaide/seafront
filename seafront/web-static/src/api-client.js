"use strict";


/**
 * The message carried by something that was thrown.
 *
 * A `catch` binding is `unknown` - JavaScript lets anything be thrown - so every handler that
 * wants to show the user what went wrong needs this one narrowing, in one place rather than
 * at each catch.
 *
 * @param {unknown} error
 * @returns {string}
 */
function errorText(error) {
    if (error instanceof Error) return error.message;
    if (typeof error === 'string') return error;
    return String(error);
}

/**
 * A config item's value, as the kind it says it is.
 *
 * `ConfigItem.value` is a union - the config system stores ints, floats, text and whole
 * objects under one field - and `value_kind` is what says which. These mirror the
 * `strvalue` / `floatvalue` / `objectvalue` accessors the server's own ConfigItem has, so
 * both sides narrow the same way instead of the browser guessing.
 *
 * @param {ConfigItem} item
 * @returns {string}
 */
function strvalue(item) {
    if (typeof item.value !== 'string') {
        throw new Error(`config ${item.handle} is ${item.value_kind}, not text`);
    }
    return item.value;
}

/**
 * @param {ConfigItem} item
 * @returns {number}
 */
function numvalue(item) {
    if (typeof item.value !== 'number') {
        throw new Error(`config ${item.handle} is ${item.value_kind}, not numeric`);
    }
    return item.value;
}

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
        this.getTargetDevice = options.getTargetDevice || (() => '');
        this.getLockToken = options.getLockToken || (() => null);
        this.ensureLease = options.ensureLease || null;
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
            } else if (response.status === 423) {
                // Hardware request without the device lease
                /** @type {LockedErrorModel} */
                const errorBody = await response.json();
                const d = errorBody.detail;
                const until = d.expires_at ? ` (until ${new Date(d.expires_at).toLocaleTimeString()})` : '';
                const userMessage = d.reason === 'not_held'
                    ? `This command needs the device lock, and nobody holds it. Acquire it from the server panel and try again.`
                    : `The microscope is locked by '${d.owner ?? 'another client'}'${until}. `
                        + `Only the lock holder can drive the hardware; take it over via the cancel dialog.`;

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
     * @param {boolean} [options.leaseRetry=true] - Whether to take the device lease and retry once if the server says nobody holds it
     * @returns {Promise<TResponse>} - Response data from server
     */
    async post(endpoint, body, options = {}) {
        const {
            context = 'API request',
            showError = true,
            leaseRetry = true,
        } = options;

        // Inject target_device (routing) and, while this client holds the device lease, lock_token
        const lockToken = this.getLockToken();
        const requestBody = {
            ...body,
            target_device: this.getTargetDevice(),
            ...(lockToken ? { lock_token: lockToken } : {}),
        };

        const url = this.buildUrl(endpoint);
        const response = await fetch(url, {
            method: 'POST',
            body: JSON.stringify(requestBody),
            headers: [['Content-Type', 'application/json']],
        });

        // Hardware commands need the lease, and this client takes one the first time it needs
        // it rather than making the user press a button first. Only `not_held` is retried:
        // a lease somebody else holds is theirs until they release it or we break it. The
        // refused request did nothing, so replaying it is safe.
        if (response.status === 423 && leaseRetry && this.ensureLease) {
            /** @type {LockedErrorModel} */
            const errorBody = await response.clone().json();
            if (errorBody.detail?.reason === 'not_held') {
                await this.ensureLease();
                return this.post(endpoint, body, { ...options, leaseRetry: false });
            }
        }

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
 * An image could not be fetched. `permanent` when the server answered that it cannot supply
 * that image (asking again for the same image will get the same answer); otherwise the
 * connection failed, and asking again may work.
 */
class ImageFetchError extends Error {
    /**
     * @param {string} message
     * @param {boolean} permanent
     */
    constructor(message, permanent) {
        super(message);
        this.name = 'ImageFetchError';
        this.permanent = permanent;
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
        this.getTargetDevice = options.getTargetDevice || (() => '');

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
     * @returns {AnySocket|null} - The socket, or null if creation failed
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
            if (existing.reconnectTimer !== null) {
                clearTimeout(existing.reconnectTimer);
                existing.reconnectTimer = null;
            }
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
                reconnectScheduled: false,
                reconnectTimer: null,
            };

            this.connections.set(name, connectionData);

            ws.onopen = () => {
                connectionData.reconnectAttempts = 0;
                connectionData.reconnectScheduled = false;
                this.onConnectionStateChange(name, true);
                // Send target_device as first message for routing validation
                const targetDevice = this.getTargetDevice();
                ws.send(JSON.stringify({ target_device: targetDevice }));
                onOpen(ws);
            };

            ws.onmessage = (event) => {
                onMessage(event);
            };

            ws.onerror = (event) => {
                this.onConnectionStateChange(name, false);
                onError(event);

                // Only schedule reconnect if this is still the current connection and not already scheduled
                const currentConn = this.connections.get(name);
                if (autoReconnect && currentConn && currentConn.ws === ws && !currentConn.reconnectScheduled) {
                    this.scheduleReconnect(name);
                }
            };

            ws.onclose = () => {
                this.onConnectionStateChange(name, false);
                onClose();

                // Only schedule reconnect if this is still the current connection and not already scheduled
                const currentConn = this.connections.get(name);
                if (autoReconnect && currentConn && currentConn.ws === ws && !currentConn.reconnectScheduled) {
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
     * @returns {AnySocket|null} - The socket, or null if not found
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

        // Avoid multiple pending timers for the same logical connection.
        if (connectionData.reconnectTimer !== null) {
            clearTimeout(connectionData.reconnectTimer);
            connectionData.reconnectTimer = null;
        }

        connectionData.reconnectScheduled = true;
        connectionData.reconnectAttempts++;

        connectionData.reconnectTimer = setTimeout(() => {
            connectionData.reconnectTimer = null;
            const currentConn = this.connections.get(name);
            // Only reconnect if this exact connection is still current.
            if (currentConn === connectionData) {
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
            if (connectionData.reconnectTimer !== null) {
                clearTimeout(connectionData.reconnectTimer);
                connectionData.reconnectTimer = null;
            }
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

export { APIClient, WebSocketManager, ImageFetchError, errorText, strvalue, numvalue };
