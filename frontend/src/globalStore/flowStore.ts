// store/users.ts
import { atom } from "nanostores";
import { Node as FlowNode, Edge as FlowEdge } from "@xyflow/react";

// id factory
const $latestNodeId = atom(0);
export const generateFlowId = () => {
  $latestNodeId.set($latestNodeId.get() + 1);
  return $latestNodeId.get().toString();
};

// nodes
export const $nodes = atom<FlowNode[]>([]);
export const setNodes = (nodes: FlowNode[]) => {
  $nodes.set(nodes);
};
export const addNode = (node: FlowNode) => {
  setNodes([...$nodes.get(), node]);
};

// edges
export const $edges = atom<FlowEdge[]>([]);
export const setEdges = (edges: FlowEdge[]) => {
  $edges.set(edges);
};
