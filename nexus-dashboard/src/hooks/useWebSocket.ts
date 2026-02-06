/**
 * useWebSocket - React hook for WebSocket connections
 *
 * Manages WebSocket lifecycle, reconnection, and message handling.
 * Used by SchematicQualityChecklist and GatherSymbolsButton for real-time updates.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface UseWebSocketOptions {
  /** Enable/disable WebSocket connection */
  enabled?: boolean;
  /** Reconnect on disconnect */
  reconnect?: boolean;
  /** Max reconnection attempts (0 = infinite) */
  maxReconnectAttempts?: number;
  /** Delay between reconnection attempts (ms) */
  reconnectDelay?: number;
  /** Connection timeout (ms) */
  connectionTimeout?: number;
  /** Callback when connection opens */
  onOpen?: () => void;
  /** Callback when connection closes */
  onClose?: (event: CloseEvent) => void;
  /** Callback when error occurs */
  onError?: (event: Event) => void;
  /** Callback when message received */
  onMessage?: (message: string) => void;
}

export interface UseWebSocketReturn {
  /** Is WebSocket connected? */
  isConnected: boolean;
  /** Last received message */
  lastMessage: string | null;
  /** Send message to WebSocket */
  send: (message: string) => void;
  /** Manually connect */
  connect: () => void;
  /** Manually disconnect */
  disconnect: () => void;
  /** Connection error */
  error: Event | null;
  /** Reconnection attempt count */
  reconnectAttempts: number;
}

/**
 * React hook for WebSocket connections with auto-reconnect
 */
export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    enabled = true,
    reconnect = true,
    maxReconnectAttempts = 5,
    reconnectDelay = 3000,
    connectionTimeout = 10000,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const [error, setError] = useState<Event | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up timeouts
  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || !url) return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      console.log(`[useWebSocket] Connecting to ${url}...`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      // Connection timeout
      connectionTimeoutRef.current = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.warn(`[useWebSocket] Connection timeout after ${connectionTimeout}ms`);
          ws.close();
        }
      }, connectionTimeout);

      // On open
      ws.onopen = () => {
        console.log(`[useWebSocket] Connected to ${url}`);
        clearTimeouts();
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
        onOpen?.();
      };

      // On message
      ws.onmessage = (event) => {
        const message = event.data;
        setLastMessage(message);
        onMessage?.(message);
      };

      // On error
      ws.onerror = (event) => {
        console.error('[useWebSocket] Error:', event);
        setError(event);
        onError?.(event);
      };

      // On close
      ws.onclose = (event) => {
        console.log(`[useWebSocket] Disconnected (code: ${event.code}, reason: ${event.reason})`);
        clearTimeouts();
        setIsConnected(false);
        wsRef.current = null;
        onClose?.(event);

        // Attempt reconnection
        if (reconnect && enabled) {
          const attempts = reconnectAttempts + 1;
          if (maxReconnectAttempts === 0 || attempts <= maxReconnectAttempts) {
            console.log(
              `[useWebSocket] Reconnecting in ${reconnectDelay}ms (attempt ${attempts}/${maxReconnectAttempts || '∞'})`
            );
            setReconnectAttempts(attempts);
            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, reconnectDelay);
          } else {
            console.warn('[useWebSocket] Max reconnection attempts reached');
          }
        }
      };
    } catch (err) {
      console.error('[useWebSocket] Failed to create WebSocket:', err);
      setError(err as Event);
    }
  }, [
    enabled,
    url,
    reconnect,
    maxReconnectAttempts,
    reconnectDelay,
    connectionTimeout,
    reconnectAttempts,
    onOpen,
    onClose,
    onError,
    onMessage,
    clearTimeouts,
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    clearTimeouts();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, [clearTimeouts]);

  // Send message
  const send = useCallback((message: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[useWebSocket] Cannot send message: WebSocket not connected');
      return;
    }
    wsRef.current.send(message);
  }, []);

  // Connect on mount/enabled change
  useEffect(() => {
    if (enabled) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [enabled, url]); // Only reconnect when enabled or url changes

  return {
    isConnected,
    lastMessage,
    send,
    connect,
    disconnect,
    error,
    reconnectAttempts,
  };
}

/**
 * Parse PROGRESS:{json} messages from Python stdout
 */
export function parseProgressMessage(message: string): any | null {
  if (!message.startsWith('PROGRESS:')) {
    return null;
  }
  try {
    return JSON.parse(message.substring(9));
  } catch (err) {
    console.error('Failed to parse progress message:', err);
    return null;
  }
}
