// store/users.ts
import { atom } from "nanostores";
import {
  Node as FlowNode,
  Edge as FlowEdge,
  ReactFlowInstance,
} from "@xyflow/react";

// interface FlowNodeWithOrigin extends FlowNode {
//   origin: number[];
// }

// id factory
const $latestNodeId = atom(0);
export const generateFlowId = () => {
  $latestNodeId.set($latestNodeId.get() + 1);
  return $latestNodeId.get().toString();
};

// flow ref
export const $flowInst = atom<
  ReactFlowInstance<FlowNode, FlowEdge> | undefined
>(undefined);

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
export const addEdge = (edge: FlowEdge) => {
  setEdges([...$edges.get(), edge]);
};
