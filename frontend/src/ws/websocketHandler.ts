import { useCallback, useEffect, useState } from "react";
import { notifError } from "../utils/notifications";

export const useWebsocket = (
  url: string,
  onMessage: (data: unknown) => void
) => {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isError, setError] = useState(false);
  const [duration, setDuration] = useState(0);

  const start = useCallback(() => {
    if (!url) {
      console.error("No url provided for websocket");
      return;
    }

    setWs(new WebSocket(url));
    setIsConnecting(true);
  }, [url]);

  const stop = useCallback(() => {
    if (!ws) return;

    ws.close();
  }, [ws]);

  useEffect(() => {
    if (!isConnected) {
      return;
    }
    setIsConnecting(false);

    const interval = setInterval(() => {
      setDuration((prev) => prev + 1000);
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [isConnected]);

  // set handlers when websocket connection changes
  useEffect(() => {
    if (!ws) return;

    ws.onerror = () => {
      notifError({
        message: "websocket connection errror",
        title: "Error",
      });
      setConnected(false);
      stop();
      setIsConnecting(false);
      setError(true);
    };

    ws.onopen = () => {
      setConnected(true);
      setIsConnecting(false);
    };

    ws.onclose = () => {
      setConnected(false);
      setIsConnecting(false);
    };

    ws.onmessage = ({ data }) => {
      console.log(data);
    };
  }, [ws, onMessage, stop]);

  const sendWsData = useCallback(
    (data: ArrayBuffer) => {
      if (ws == null || ws.readyState != ws.OPEN || !isConnected) return;
      ws.send(data);
    },
    [ws, isConnected]
  );

  const connected = isConnected && !!ws;

  return {
    sendWsData,
    stop,
    start,
    connected,
    duration,
    isConnecting,
    isError,
  };
};
