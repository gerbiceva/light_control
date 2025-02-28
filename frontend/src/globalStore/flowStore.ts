// store/users.ts
import { atom, computed } from "nanostores";
import { ReactFlowInstance } from "@xyflow/react";
import { persistentAtom } from "@nanostores/persistent";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { mainFlow, SubGraph } from "../subgraph/Subgraph";

// function for storing and reading serialized data
const parser = {
  encode: JSON.stringify,
  decode: JSON.parse,
};

// id factory
const $latestNodeId = persistentAtom<number>("flowIdManager", 0, parser);

export const generateFlowId = () => {
  $latestNodeId.set($latestNodeId.get() + 1);
  return $latestNodeId.get().toString();
};

export interface AppState {
  subgraphs: SubGraph[];
  currentSubgraphId: number;
}
export const $appState = atom<AppState>({
  subgraphs: [
    {
      id: 0,
      name: "main",
      description: "This main flow gets executed always",
      nodes: [],
      edges: [],
    },
    {
      id: 1,
      name: "sub",
      description: "This main flow gets executed always",
      nodes: [],
      edges: [],
    },
    {
      id: 2,
      name: "sub2",
      description: "This main flow gets executed always",
      nodes: [],
      edges: [],
    },
  ],
  currentSubgraphId: 0,
});

export const setActiveSubgraph = (id: number) => {
  $appState.set({
    ...$appState.get(),
    currentSubgraphId: id,
  });
};

// flow ref
export const $flowInst = atom<
  ReactFlowInstance<CustomFlowNode, CustomFlowEdge> | undefined
>(undefined);

// nodes
// export const $nodes = persistentAtom<CustomFlowNode[]>("nodes", [], parser);
export const $nodes = computed($appState, () => {
  console.log($appState.get());
  return $appState.get().subgraphs[$appState.get().currentSubgraphId].nodes;
});

export const setNodes = (nodes: CustomFlowNode[]) => {
  const newState = { ...$appState.get() };
  newState.subgraphs[newState.currentSubgraphId].nodes = nodes;
  $appState.set(newState);
};
export const addNode = (node: CustomFlowNode) => {
  setNodes([node, ...$nodes.get()]);
};

// edges
export const $edges = computed(
  $appState,
  () => $appState.get().subgraphs[$appState.get().currentSubgraphId].edges
);
export const setEdges = (edges: CustomFlowEdge[]) => {
  const newState = $appState.get();
  newState.subgraphs[newState.currentSubgraphId].edges = edges;
  $appState.set(newState);
  console.log({ newState });
};

export const resetState = () => {
  $appState.set({
    subgraphs: [mainFlow],
    currentSubgraphId: 0,
  });
};

export const getSaveFile = (): AppState => {
  return $appState.get();
};
