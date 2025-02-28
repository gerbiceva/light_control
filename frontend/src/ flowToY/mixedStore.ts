interface SyncPos {
  x: number;
  y: number;
}

interface SyncNode {
  state: "working" | "error" | "idle";
  namespace: string;
  name: string;
  value?: string;
  id: string;
  position: SyncPos;
}

interface SyncEdge {
  fromNodeId: string;
  fromNodeHandle: string;
  toNodeId: string;
  toNodeHandle: string;
}

interface SyncGraph {
  name: string;
  description?: string;
  nodes: SyncNode[];
  edges: SyncEdge[];
}

interface SyncedStore {
  mainGraph: SyncGraph;
  subGraphs: SyncGraph[];
}
