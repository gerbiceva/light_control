import { useStore } from "@nanostores/react";
import { $edges, $nodes } from "../globalStore/flowStore";
import { useCallback, useEffect } from "react";
import { useDebouncedValue } from "@mantine/hooks";
import { addSyncPromise, changeHappened } from "../globalStore/loadingStore";
import { sync } from "./syncFunc";

const updateIntervalMs = 500;

export const useSync = () => {
  const edges = useStore($edges);
  const nodes = useStore($nodes);

  const [debouncedEdges] = useDebouncedValue(edges, updateIntervalMs);
  const [debouncedNodes] = useDebouncedValue(nodes, updateIntervalMs);

  const update = useCallback(() => {
    addSyncPromise(
      new Promise((resolve) => {
        setTimeout(() => {
          resolve(0);
        }, 300);
      })
    );
  }, []);

  // const update = useCallback(() => {
  //   addSyncPromise(sync(nodes, edges));
  // }, [edges, nodes]);

  useEffect(() => {
    changeHappened();
  }, [edges, nodes]);

  useEffect(() => {
    update();
  }, [debouncedEdges, debouncedNodes, update]);
};
