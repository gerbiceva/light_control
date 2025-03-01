// store/users.ts
import { persistentAtom } from "@nanostores/persistent";
import { ReactFlowInstance } from "@xyflow/react";
import { atom } from "nanostores";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { SubGraph } from "../subgraph/Subgraph";

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
  main: SubGraph;
}

// flow ref
export const $flowInst = atom<
  ReactFlowInstance<CustomFlowNode, CustomFlowEdge> | undefined
>(undefined);
