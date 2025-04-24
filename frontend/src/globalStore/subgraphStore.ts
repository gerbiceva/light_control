import { useStore } from "@nanostores/react";
import { atom } from "nanostores";
import { useCallback, useMemo } from "react";
import { SubGraph } from "../components/Subgraph/Subgraph";
import { generateGraphId } from "./flowStore";
import { $syncedAppState } from "../crdt/globalSync";
import { addSubgraph } from "../crdt/repo";

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
    if (!syncedStore) return undefined;

    return activeGraph === "main"
      ? syncedStore.main
      : syncedStore.subgraphs[activeGraph];
  }, [activeGraph, syncedStore]);

  // useEffect(() => {
  //   console.log({ activeGraphObj });
  // }, [activeGraphObj]);

  const setActiveGraph = useCallback((graph: "main" | number) => {
    $subgraphPages.set({
      ...$subgraphPages.get(),
      activeGraph: graph,
    });
  }, []);

  const newSubGraph = useCallback(
    (name: string, description?: string) => {
      const newGraph = createEmptySubgraph(name, description);
      addSubgraph(newGraph);
      addVisibleSubgraph(newGraph.id);
      setActiveGraph(newGraph.id);
      return newGraph;
    },
    [setActiveGraph]
  );

  const visibleGraphObj = useMemo(() => {
    if (!syncedStore) return [];

    return visibleGraphs
      .map((id) => syncedStore.subgraphs[id])
      .filter((graph) => graph !== undefined);
  }, [visibleGraphs, syncedStore]);

  return {
    activeGraph: activeGraphObj,
    setActiveGraph,
    newSubGraph,
    visibleGraphs: visibleGraphObj,
  };
};
