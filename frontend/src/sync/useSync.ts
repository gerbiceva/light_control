import { useStore } from "@nanostores/react";
import { $edges, $nodes } from "../globalStore/flowStore";
import { useEffect } from "react";
import { useDebouncedValue } from "@mantine/hooks";
import { addSyncPromise, changeHappened } from "../globalStore/loadingStore";

const updateIntervalMs = 500;

export const useSync = () => {
  const edges = useStore($edges);
  const nodes = useStore($nodes);

  const [debouncedEdges] = useDebouncedValue(edges, updateIntervalMs);
  const [debouncedNodes] = useDebouncedValue(nodes, updateIntervalMs);

  const update = () => {
    console.log("changeee");

    addSyncPromise(
      new Promise((resolve) => {
        setTimeout(() => {
          resolve(0);
        }, 300);
      })
    );
  };

  useEffect(() => {
    changeHappened();
  }, [edges, nodes]);

  useEffect(() => {
    update();
  }, [debouncedEdges, debouncedNodes]);
};
