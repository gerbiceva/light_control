import * as Y from "yjs";
import { WebsocketProvider } from "y-websocket";
import { AppState } from "../globalStore/flowStore";
import {
  applyEdgeChanges,
  applyNodeChanges,
  EdgeChange,
  NodeChange,
} from "@xyflow/react";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { atom } from "nanostores";
import { SubGraph } from "../components/Subgraph/Subgraph";
import { $subgraphPages } from "../globalStore/subgraphStore";

export const yAppState = new Y.Doc();
// const yarray = yAppState.getArray("count");
const roomName = "app-state";
const port = 42069;
const YSyncStore = yAppState.getMap("syncedAppState"); // dont change the name

const websocketProvider = new WebsocketProvider(
  `ws:localhost:${port}`,
  roomName,
  yAppState
);

websocketProvider.on("status", (wsStatusEv) => {
  console.log({ ev: wsStatusEv });
});
websocketProvider.on("sync", (wsSyncEv) => {
  console.log({ ev: wsSyncEv });
});

export const updateState = () => {
  setYState(getYState());
};

// type safety for this thing is non existent...just ...trust me
export const getYState = () => {
  return YSyncStore.toJSON().state as AppState;
};

// type safety for this thing is non existent...just ...trust me
export const setYState = (state: AppState) => {
  YSyncStore.set("state", { ...state });
};

export const getActiveYgraph = () => {
  const activeGraphId = $subgraphPages.get().activeGraph;
  if (activeGraphId == "main") {
    return getYState().main;
  }
  const g = getYState().subgraphs.find((graph) => graph.id == activeGraphId)!;
  return g;
};

export const addNode = (node: CustomFlowNode) => {
  const g = getActiveYgraph();
  g.nodes.push(node);
  updateState();
};

export const setEdges = (edges: CustomFlowEdge[]) => {
  const g = getActiveYgraph();
  g.edges = edges;
  updateState();
};

export const setNodes = (nodes: CustomFlowNode[]) => {
  const g = getActiveYgraph();
  g.nodes = nodes;
  updateState();
};

export const $syncedAppState = atom<AppState>(getYState());
YSyncStore.observe(() => {
  $syncedAppState.set(getYState());
});

export const setSubgraphs = (subgraphs: SubGraph[]) => {
  getYState().subgraphs = subgraphs;
  updateState();
};

export const onNodesChange = (change: NodeChange<CustomFlowNode>[]) => {
  const g = getActiveYgraph();
  g.nodes = applyNodeChanges<CustomFlowNode>(change, g.nodes);
  updateState();
};

export const onEdgesChange = (changes: EdgeChange<CustomFlowEdge>[]) => {
  const g = getActiveYgraph();
  g.edges = applyEdgeChanges<CustomFlowEdge>(changes, g.edges);
  updateState();
};
