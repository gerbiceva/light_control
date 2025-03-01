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

websocketProvider.on("status", (ev) => {
  console.log({ ev });
});

export const updateState = () => {
  setYState(getYState());
};

export const getYState = () => {
  return YSyncStore.toJSON().state as AppState;
};

export const setYState = (state: AppState) => {
  YSyncStore.set("state", { ...state });
};
export const addNode = (node: CustomFlowNode) => {
  const curr = getYState();
  curr.main.nodes.push(node);
  setYState(curr);
};

export const setEdges = (edges: CustomFlowEdge[]) => {
  getYState().main.edges = edges;
  updateState();
};

export const setNodes = (nodes: CustomFlowNode[]) => {
  getYState().main.nodes = nodes;
  updateState();
};

// setYState({
//   main: {
//     edges: [],
//     nodes: [],
//     name: "main",
//     id: 99,
//   },
//   subgraphs: [],
// });

export const $syncedAppState = atom<AppState>(getYState());
YSyncStore.observe(() => {
  console.log(getYState());
  $syncedAppState.set(getYState());
});

export const onNodesChange = (change: NodeChange<CustomFlowNode>[]) => {
  getYState().main.nodes = applyNodeChanges<CustomFlowNode>(
    change,
    getYState().main.nodes
  );
  updateState();
};
export const onEdgesChange = (changes: EdgeChange<CustomFlowEdge>[]) => {
  getYState().main.edges = applyEdgeChanges(changes, getYState().main.edges);
  updateState();
};
