import { Box } from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useState, useCallback, useEffect } from "react";
import {
  ReactFlow,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type FitViewOptions,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type OnNodeDrag,
  type NodeTypes,
  type DefaultEdgeOptions,
  Background,
  Controls,
  MiniMap,
} from "@xyflow/react";
import { ColorNode } from "../flow/Nodes/ColorNode";
import { IntNode } from "../flow/Nodes/IntNode";
import { FloatNode } from "../flow/Nodes/FloatNode";
import { StringNode } from "../flow/Nodes/StringNode";
import { CurveNode } from "../flow/Nodes/CurveNode";
import { GenerateComputeNodeFromCapability } from "../flow/Nodes/ComputeNodes/ComputeNodeFactory";
import { testingCapabilityNode } from "../flow/Nodes/ComputeNodes/test";

const nodeTypes: NodeTypes = {
  colorNode: ColorNode,
  intNode: IntNode,
  floatNode: FloatNode,
  stringNode: StringNode,
  curveNode: CurveNode,
  [testingCapabilityNode.name]: GenerateComputeNodeFromCapability(
    testingCapabilityNode
  ),
};
const initialNodes: Node[] = [
  { id: "1", data: { label: "Node 1" }, position: { x: 5, y: 5 } },
  {
    id: "3",
    position: { x: 100, y: 10 },
    data: {
      color: "hsl(78, 45%, 72%)",
    },
    type: "colorNode",
  },
  {
    id: "4",
    position: { x: 200, y: 10 },
    data: {
      int: 0,
    },
    type: "intNode",
  },
  {
    id: "5",
    position: { x: 300, y: 10 },
    data: {
      float: 0.0,
    },
    type: "floatNode",
  },
  {
    id: "6",
    position: { x: 300, y: 10 },
    data: {
      str: "nekaj ojla",
    },
    type: "stringNode",
  },
  {
    id: "7",
    position: { x: 300, y: 10 },
    data: {
      points: [
        { x: 0, y: 0 },
        { x: 0.25, y: 0.25 },
        { x: 0.75, y: 0.75 },
        { x: 1, y: 1 },
      ],
    },
    type: "curveNode",
  },
  {
    id: "8",
    position: { x: 300, y: 10 },
    data: {},
    type: testingCapabilityNode.name,
  },
];

const initialEdges: Edge[] = [{ id: "e1-2", source: "1", target: "2" }];

const fitViewOptions: FitViewOptions = {
  padding: 0.5,
};

const defaultEdgeOptions: DefaultEdgeOptions = {
  animated: true,
};

const onNodeDrag: OnNodeDrag = (_, node) => {
  console.log("drag event", node.data);
};

export const NodeView = () => {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);

  useEffect(() => {
    console.log(initialNodes);
  }, []);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );
  const onConnect: OnConnect = useCallback(
    (connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  return (
    <Box w="100%" h="100%" pos="relative">
      <ReactFlow
        nodes={nodes}
        nodeTypes={nodeTypes}
        edges={edges}
        // edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDrag={onNodeDrag}
        fitView
        fitViewOptions={fitViewOptions}
        defaultEdgeOptions={defaultEdgeOptions}
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </Box>
  );
};
