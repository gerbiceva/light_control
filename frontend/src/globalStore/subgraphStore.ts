import { useStore } from "@nanostores/react";
import { atom } from "nanostores";
import { $syncedAppState, setSubgraphs } from "../crdt/repo";
import { useCallback, useMemo } from "react";
import { SubGraph } from "../components/Subgraph/Subgraph";
import { generateGraphId } from "./flowStore";

interface SubgraphPageStore {
  activeGraph: "main" | number;
  visibleGraphs: number[];
}

export const $subgraphPages = atom<SubgraphPageStore>({
  activeGraph: "main",
  visibleGraphs: [],
});

export const createEmptySubgraph = (
  name: string,
  description?: string
): SubGraph => {
  return {
    id: generateGraphId(),
    edges: [],
    nodes: [],
    description: description,
    name: name,
  };
};

export const setVisibleSubgraphs = (graphs: number[]) => {
  $subgraphPages.set({
    ...$subgraphPages.get(),
    visibleGraphs: graphs,
  });
};

export const addVisibleSubgraph = (graph: number) => {
  setVisibleSubgraphs([...$subgraphPages.get().visibleGraphs, graph]);
};

export const useSubgraphs = () => {
  const { activeGraph, visibleGraphs } = useStore($subgraphPages);
  const syncedStore = useStore($syncedAppState);

  const activeGraphObj = useMemo(() => {
    if (syncedStore == undefined) {
      return;
    }
    if (activeGraph == "main") {
      return syncedStore.main;
    }
    return syncedStore.subgraphs.find((graph) => graph.id == activeGraph);
  }, [activeGraph, syncedStore]);

  const setActiveGraph = useCallback((activeGraph: "main" | number) => {
    $subgraphPages.set({
      ...$subgraphPages.get(),
      activeGraph: activeGraph,
    });
  }, []);

  const newSubGraph = useCallback(
    (name: string, description?: string) => {
      const newGraph = createEmptySubgraph(name, description);
      setSubgraphs([newGraph, ...syncedStore.subgraphs]);
      addVisibleSubgraph(newGraph.id);
      setActiveGraph(newGraph.id);
    },
    [setActiveGraph, syncedStore]
  );

  return {
    activeGraph: activeGraphObj,
    setActiveGraph,
    newSubGraph,
    visibleGraphs: syncedStore
      ? syncedStore.subgraphs.filter((graph) =>
          visibleGraphs.includes(graph.id)
        )
      : [],
  };
};
