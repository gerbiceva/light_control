import { useCallback, useEffect, useState } from "react";
import { notifError } from "../utils/notifications";

export const useWebsocket = (
  url: string,
  onMessage: (data: unknown) => void,
) => {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isError, setError] = useState(false);

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
    setIsConnecting(false);
  }, [ws]);

  useEffect(() => {
    if (!isConnected) {
      return;
    }
    setIsConnecting(false);
  }, [isConnected]);

  useEffect(() => {
    start();
  }, [start]);

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
      onMessage(data);
    };
  }, [ws, onMessage, stop]);

  const sendWsData = useCallback(
    (data: ArrayBuffer) => {
      if (ws == null || ws.readyState != ws.OPEN || !isConnected) return;
      ws.send(data);
    },
    [ws, isConnected],
  );

  const connected = isConnected && !!ws;

  return {
    sendWsData,
    stop,
    start,
    connected,
    isConnecting,
    isError,
  };
};
