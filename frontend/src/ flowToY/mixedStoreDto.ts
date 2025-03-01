import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";

export interface Ypos {
  x: number;
  y: number;
}

export interface YNode {
  state: "working" | "error" | "idle";
  namespace: string;
  name: string;
  value?: string;
  id: string;
  position: Ypos;
}

export interface YEdge {
  fromNodeId: string;
  fromNodeHandle: string;
  toNodeId: string;
  toNodeHandle: string;
}

export interface YGraph {
  name: string;
  description?: string;
  nodes: YNode[];
  edges: YEdge[];
}

export interface SyncedStore {
  mainGraph: YGraph;
  subGraphs: YGraph[];
}

export interface MainGraph {
  name: string;
  description?: string;
  nodes: CustomFlowNode[];
  edges: CustomFlowEdge[];
}

export interface MainStore {
  mainGraph: MainGraph;
  subGraphs: MainGraph[];
}
