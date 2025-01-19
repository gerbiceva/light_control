// store/users.ts
import { atom } from "nanostores";
import {
  Node as FlowNode,
  Edge as FlowEdge,
  ReactFlowInstance,
} from "@xyflow/react";
import { persistentAtom } from "@nanostores/persistent";

// id factory
const $latestNodeId = persistentAtom<number>("flowIdManager", 0, {
  encode: JSON.stringify,
  decode: JSON.parse,
});

export const generateFlowId = () => {
  $latestNodeId.set($latestNodeId.get() + 1);
  return $latestNodeId.get().toString();
};

// flow ref
export const $flowInst = atom<
  ReactFlowInstance<FlowNode, FlowEdge> | undefined
>(undefined);

// nodes
export const $nodes = persistentAtom<FlowNode[]>("nodes", [], {
  encode: JSON.stringify,
  decode: JSON.parse,
});
export const setNodes = (nodes: FlowNode[]) => {
  $nodes.set(nodes);
};
export const addNode = (node: FlowNode) => {
  setNodes([...$nodes.get(), node]);
};

// edges
export const $edges = persistentAtom<FlowEdge[]>("edges", [], {
  encode: JSON.stringify,
  decode: JSON.parse,
});
export const setEdges = (edges: FlowEdge[]) => {
  $edges.set(edges);
};
export const addEdge = (edge: FlowEdge) => {
  setEdges([...$edges.get(), edge]);
};

export const resetState = () => {
  setEdges([]);
  setNodes([]);
};

export interface SaveFile {
  nodes: FlowNode[];
  edges: FlowEdge[];
}
export const getSaveFile = (): SaveFile => {
  return {
    edges: $edges.get(),
    nodes: $nodes.get(),
  };
};
