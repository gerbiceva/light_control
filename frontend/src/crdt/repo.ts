import * as Y from "yjs";
import { AppState } from "../globalStore/flowStore";
import { EdgeChange, NodeChange } from "@xyflow/react";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { SubGraph } from "../components/Subgraph/Subgraph";
import { $subgraphPages } from "../globalStore/subgraphStore";
import { WebsocketProvider } from "y-websocket";
import { toShallowYMap } from "../ flowToY/syncUtils";

export const yAppState = new Y.Doc();
export const roomName = "app-state";
export const port = 42069;
export const transactionName = "client-flow-update-origin";
const YSyncStore = yAppState.getMap("syncedAppState"); // dont change the name

export const websocketProvider = new WebsocketProvider(
  `ws:${window.location.hostname}:${port}`,
  roomName,
  yAppState,
  {
    connect: true,
  }
);

websocketProvider.on("status", (wsStatusEv) => {
  console.log({ ev: wsStatusEv });
});
websocketProvider.on("sync", (wsSyncEv) => {
  console.log({ ev: wsSyncEv });
});

// Add to repo.ts
export const waitForSync = (): Promise<void> => {
  return new Promise((resolve) => {
    if (websocketProvider.synced) {
      resolve();
    } else {
      const handler = () => {
        websocketProvider.off("sync", handler);
        resolve();
      };
      websocketProvider.on("sync", handler);
    }
  });
};

const clientTransaction = (transactionFunc: () => void) => {
  Y.transact(yAppState, transactionFunc, transactionName);
};

yAppState.on("update", (_, __, doc) => {
  console.log(doc.toJSON());
});

export const initializeYState = async (): Promise<boolean> => {
  try {
    await waitForSync();
    return true;
  } catch (error) {
    console.error("Failed to initialize Yjs state:", error);
    return false;
  }
};

export const getYState2 = () => {
  const st = (YSyncStore.get("state") as Y.Map<AppState>) || undefined;
  return st;
};

export const getActiveYgraph2 = (): Y.Map<SubGraph> | undefined => {
  const st = getYState2();
  if (!st) {
    console.error("Yjs state not initialized");
    return undefined;
  }

  const activeGraphId = $subgraphPages.get().activeGraph;
  if (!activeGraphId) {
    console.error("No active graph ID found");
    return undefined;
  }

  try {
    if (activeGraphId === "main") {
      return st.get("main") as Y.Map<SubGraph> | undefined;
    }

    const subgraphs = st.get("subgraphs") as Y.Map<Y.Map<SubGraph>> | undefined;
    if (!subgraphs) {
      console.error("Subgraphs collection not found");
      return undefined;
    }

    return subgraphs.get(activeGraphId.toString());
  } catch (error) {
    console.error("Error accessing Yjs graph data:", error);
    return undefined;
  }
};

export const getActiveYgraph = () => {
  const g = getActiveYgraph2();
  if (!g) {
    return;
  }
  console.log({ g });

  return (g.toJSON() as SubGraph) || undefined;
};

export const addSubgraph = (subgraph: SubGraph) => {
  clientTransaction(() => {
    const st = getYState2();
    if (!st) {
      console.error("Yjs state not initialized");
      return;
    }

    const subgraphs = st.get("subgraphs") as Y.Map<Y.Map<SubGraph>> | undefined;
    if (!subgraphs) {
      throw new Error("No subgraph structure found");

      // subgraphs = new Y.Map();
      // st.set("subgraphs", subgraphs);
    }

    const subgraphMap = new Y.Map();
    subgraphMap.set("id", subgraph.id);
    subgraphMap.set("name", subgraph.name);
    subgraphMap.set("description", subgraph.description);

    const nodes = new Y.Array();
    const edges = new Y.Array();

    // Only push items if they exist
    if (subgraph.nodes) {
      nodes.push(subgraph.nodes);
    }
    if (subgraph.edges) {
      edges.push(subgraph.edges);
    }

    subgraphMap.set("nodes", nodes);
    subgraphMap.set("edges", edges);

    subgraphs.set(subgraph.id.toString(), subgraphMap as Y.Map<SubGraph>);
    console.log("Subgraph added:", subgraph.id);
  });
};

export const addEdge = (edge: CustomFlowEdge | undefined) => {
  if (!edge) {
    return;
  }
  clientTransaction(() => {
    const g = getActiveYgraph2();
    if (!g) return;

    const yEdges = g.get("edges") as Y.Array<CustomFlowEdge> | undefined;
    if (!yEdges) return;

    const currentEdges = yEdges.toArray();
    const existingIndex = currentEdges.findIndex((e) => e.id === edge.id);

    if (existingIndex !== -1) {
      yEdges.delete(existingIndex, 1);
    }
    yEdges.push([edge]);
  });
};

const processSingleChange = (chage: NodeChange<CustomFlowNode>) => {
  const g = getActiveYgraph2();
  if (!g) return;

  const yNodes = g.get("nodes") as Y.Array<Y.Map<CustomFlowNode>> | undefined;
  if (!yNodes) return;
  const change = chage;
  const currentNodes = yNodes
    .toArray()
    .map((node) => node.toJSON()) as CustomFlowNode[];
  switch (change.type) {
    case "add": {
      console.log("added", change.item);
      yNodes.push([toShallowYMap(change.item)]);
      return;
    }

    case "remove": {
      const removeIndex = currentNodes.findIndex((n) => n.id === change.id);
      if (removeIndex !== -1) yNodes.delete(removeIndex, 1);
      console.log("removed", removeIndex);
      return;
    }

    // case "dimensions":
    // case "select":
    case "position": {
      const updateIndex = currentNodes.findIndex((n) => n.id === change.id);
      if (updateIndex !== -1) {
        // const updatedNode = {
        //   ...currentNodes[updateIndex],
        //   ...("position" in change ? { position: change.position } : {}),
        //   // ...("dimensions" in change ? { dimensions: change.dimensions } : {}),
        //   // ...("selected" in change ? { selected: change.selected } : {}),
        // };

        // console.log("change", change.type, updatedNode);
        const nodeToUpdate = yNodes.get(updateIndex);

        //@ts-expect-error for Yjs types are plain wrong
        nodeToUpdate.set("position", change.position);
      }
      return;
    }

    case "replace": {
      const replaceIndex = currentNodes.findIndex((n) => n.id === change.id);
      if (replaceIndex !== -1) {
        // First delete the old node
        yNodes.delete(replaceIndex, 1);
        // Then add the new node at the same position
        yNodes.insert(replaceIndex, [toShallowYMap(change.item)]);
      }
      return;
    }
  }
  return;
};

export const onNodesChange = (changes: NodeChange<CustomFlowNode>[]) => {
  clientTransaction(() => {
    const g = getActiveYgraph2();
    if (!g) return;

    const yNodes = g.get("nodes") as Y.Array<CustomFlowNode> | undefined;
    if (!yNodes) return;

    if (!changes.length) return;

    changes.map((change) => {
      processSingleChange(change);
    });
  });
};

const processSingleEdgeChange = (change: EdgeChange<CustomFlowEdge>) => {
  const g = getActiveYgraph2();
  if (!g) return;

  const yEdges = g.get("edges") as Y.Array<CustomFlowEdge> | undefined;
  if (!yEdges) return;

  const currentEdges = yEdges.toArray();

  switch (change.type) {
    case "add": {
      console.log("edge added", change.item);
      yEdges.push([change.item]);
      return;
    }

    case "remove": {
      const removeIndex = currentEdges.findIndex((e) => e.id === change.id);
      if (removeIndex !== -1) {
        yEdges.delete(removeIndex, 1);
        console.log("edge removed", removeIndex);
      }
      return;
    }

    case "select": {
      const updateIndex = currentEdges.findIndex((e) => e.id === change.id);
      if (updateIndex !== -1) {
        const updatedEdge = {
          ...currentEdges[updateIndex],
          selected: change.selected,
        };
        console.log("edge selection changed", updatedEdge);
        yEdges.delete(updateIndex, 1);
        yEdges.insert(updateIndex, [updatedEdge]);
      }
      return;
    }

    case "replace": {
      const replaceIndex = currentEdges.findIndex((e) => e.id === change.id);
      if (replaceIndex !== -1) {
        yEdges.delete(replaceIndex, 1);
        yEdges.insert(replaceIndex, [change.item]);
        console.log("edge replaced", change.item);
      }
      return;
    }
  }
};

export const onEdgesChange = (changes: EdgeChange<CustomFlowEdge>[]) => {
  clientTransaction(() => {
    const g = getActiveYgraph2();
    if (!g) return;

    const yEdges = g.get("edges") as Y.Array<CustomFlowEdge> | undefined;
    if (!yEdges) return;

    if (!changes.length) return;

    changes.forEach((change) => {
      processSingleEdgeChange(change);
    });
  });
};

// Reusing your existing operations:
export const addNode = (node: CustomFlowNode) => {
  clientTransaction(() => {
    const g = getActiveYgraph2();
    if (!g) {
      console.error("no active graph, cant find node");
      return;
    }

    const nodes = g.get("nodes") as Y.Array<Y.Map<CustomFlowNode>> | undefined;
    if (!nodes) {
      console.error("no nodes key, cant complete node push");
      return;
    }

    nodes.push([toShallowYMap(node)]);
  });
};
