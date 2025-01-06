import { Box } from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useCallback, useEffect } from "react";
import {
  ReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type FitViewOptions,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type NodeTypes,
  type DefaultEdgeOptions,
  Background,
  Controls,
  MiniMap,
} from "@xyflow/react";
import { getComputedNodes } from "../flow/Nodes/ComputeNodes/getComputedNodes";
import { $edges, $nodes, setEdges, setNodes } from "../globalStore/flowStore";
import { useStore } from "@nanostores/react";
import { inputNodes } from "../flow/Nodes/BaseNodes/utils/RegisterNodes";
import { useSync } from "../sync/useSync";
import { addColoredEdge } from "../flow/Edges/addColoredEdge";
import { isValidConnection } from "../flow/Edges/isValidCOnnection";
import { testingCapabilityNode } from "../flow/Nodes/ComputeNodes/test";

const nodeTypes: NodeTypes = {
  ...inputNodes,
  ...getComputedNodes(), // get all custom nodes, computed from capabilities
};
const initialNodes: Node[] = [
  {
    id: "3",
    position: { x: 100, y: 10 },
    data: {
      color: "hsl(78, 45%, 72%)",
    },
    type: "Color",
  },
  {
    id: "4",
    position: { x: 200, y: 10 },
    data: {
      int: 0,
    },
    type: "Int",
  },

  {
    id: "8",
    position: { x: 300, y: 10 },
    data: {},
    type: testingCapabilityNode.id,
  },
];

const initialEdges: Edge[] = [];

const fitViewOptions: FitViewOptions = {
  padding: 0.5,
};

const defaultEdgeOptions: DefaultEdgeOptions = {
  animated: true,
};

// const onNodeDrag: OnNodeDrag = (_, node) => {
//   console.log("drag event", node.data);
// };

export const NodeView = () => {
  const nodes = useStore($nodes);
  const edges = useStore($edges);
  useSync(); // sync backend

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes(applyNodeChanges(changes, nodes)),
    [nodes]
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges(applyEdgeChanges(changes, edges)),
    [edges]
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges(addColoredEdge(connection, edges));
      // console.log({ edges });
    },
    [edges]
  );

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, []);

  return (
    <Box w="100%" h="100%" pos="relative">
      <ReactFlow
        nodes={nodes}
        nodeTypes={nodeTypes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        // onNodeDrag={onNodeDrag}
        fitView
        fitViewOptions={fitViewOptions}
        defaultEdgeOptions={defaultEdgeOptions}
        isValidConnection={isValidConnection}
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </Box>
  );
};
