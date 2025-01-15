import { Box } from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type FitViewOptions,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type NodeTypes,
  type DefaultEdgeOptions,
  Background,
  Controls,
  MiniMap,
  useReactFlow,
} from "@xyflow/react";
import { getComputedNodes } from "../flow/Nodes/ComputeNodes/getComputedNodes";
import {
  $edges,
  $flowInst,
  $nodes,
  setEdges,
  setNodes,
} from "../globalStore/flowStore";
import { useStore } from "@nanostores/react";
import { inputNodes } from "../flow/Nodes/BaseNodes/utils/RegisterNodes";
import { useSync } from "../sync/useSync";
import { addColoredEdge } from "../flow/Edges/addColoredEdge";
import { isValidConnection } from "../flow/Edges/isValidConnection";
import { $capabilities } from "../globalStore/capabilitiesStore";
import { addInputOnEdgeDrop } from "../flow/Nodes/BaseNodes/utils/addInputOnEdgeDrop";
import { mapPrimitivesToNamespaced } from "../sync/namespaceUtils";

const fitViewOptions: FitViewOptions = {
  padding: 3,
};

const defaultEdgeOptions: DefaultEdgeOptions = {
  animated: true,
};

export const NodeView = () => {
  const nodes = useStore($nodes);
  const edges = useStore($edges);
  const caps = useStore($capabilities);
  const reactFlowInst = useReactFlow();

  useEffect(() => {
    console.log("inst change");
    $flowInst.set(reactFlowInst);
  }, [reactFlowInst]);

  useSync(); // sync backend

  const nodeTypes: NodeTypes = useMemo(
    () => {
      const caps = {
        ...getComputedNodes(), // get all custom nodes, computed from capabilities
        ...mapPrimitivesToNamespaced(inputNodes),
      };
      console.log({ caps });

      return caps;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [caps]
  );

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes(applyNodeChanges(changes, nodes)),
    [nodes]
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges(applyEdgeChanges(changes, edges)),
    [edges]
  );

  const onConnect: OnConnect = useCallback((connection) => {
    setEdges(addColoredEdge(connection));
  }, []);

  return (
    <Box w="100%" h="100%" pos="relative">
      <ReactFlow
        nodes={nodes}
        nodeTypes={nodeTypes}
        edges={edges}
        onConnectEnd={addInputOnEdgeDrop}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        fitViewOptions={fitViewOptions}
        defaultEdgeOptions={defaultEdgeOptions}
        isValidConnection={isValidConnection}
        deleteKeyCode={"Delete"}
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </Box>
  );
};
