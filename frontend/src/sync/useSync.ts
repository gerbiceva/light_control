import { useStore } from "@nanostores/react";
import { $edges, $nodes } from "../globalStore/flowStore";
import { useCallback, useEffect } from "react";
import { useDebouncedValue } from "@mantine/hooks";
import {
  $sync,
  addSyncPromise,
  changeHappened,
} from "../globalStore/loadingStore";
import { sync } from "./syncFunc";

const updateIntervalMs = 500;

export const useSync = () => {
  const edges = useStore($edges);
  const nodes = useStore($nodes);
  const { autoUpdate } = useStore($sync);
  const [debouncedEdges] = useDebouncedValue(edges, updateIntervalMs);
  const [debouncedNodes] = useDebouncedValue(nodes, updateIntervalMs);

  // const update = useCallback(() => {
  //   addSyncPromise(
  //     new Promise((resolve) => {
  //       setTimeout(() => {
  //         resolve(0);
  //       }, 300);
  //     })
  //   );
  // }, []);

  const update = useCallback(() => {
    if (autoUpdate) {
      addSyncPromise(sync(nodes, edges));
    }
  }, [debouncedNodes, debouncedEdges, autoUpdate]);

  useEffect(() => {
    changeHappened();
  }, [edges, nodes]);

  useEffect(() => {
    update();
  }, [debouncedEdges, debouncedNodes, update]);
};
