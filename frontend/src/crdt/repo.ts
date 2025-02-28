import * as Y from "yjs";
import { WebsocketProvider } from "y-websocket";
import { useCallback, useEffect, useState } from "react";

const yAppState = new Y.Doc();
const yarray = yAppState.getArray("count");
const roomName = "app-state";
const port = 42069;

export const useTask = () => {
  const [num, setNum] = useState<number>(0);

  useEffect(() => {
    // array of numbers which produce a sum
    const observer = () => {
      // print updates when the data changes
      // console.log("new sum: " + yarray.toArray().reduce((a, b) => a + b));
      setNum((yarray.toArray() as number[]).reduce((a, b) => a + b));
    };
    // observe changes of the sum
    // Sync clients with the y-websocket provider
    const websocketProvider = new WebsocketProvider(
      `ws:localhost:${port}`,
      roomName,
      yAppState
    );

    yarray.observe(observer);

    return () => {
      yarray.unobserve(observer);
      websocketProvider.disconnect();
    };
  }, []);

  const p = useCallback((num: number) => {
    yarray.push([num]); // => "new sum: 1"
  }, []);

  return {
    num,
    p,
  };
};
