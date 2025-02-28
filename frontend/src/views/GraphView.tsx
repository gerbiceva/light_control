import { Box, ColorSwatch, Stack } from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useDisclosure } from "@mantine/hooks";
import { useStore } from "@nanostores/react";
import {
  Background,
  Controls,
  Edge,
  MiniMap,
  NodeChange,
  ReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  type DefaultEdgeOptions,
  type FitViewOptions,
  type NodeTypes,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Point } from "react-bezier-spline-editor/core";
import { GroupContextMenu } from "../components/GroupContextMenu";
import { addColoredEdge } from "../flow/Edges/addColoredEdge";
import { isValidConnection } from "../flow/Edges/isValidCOnnection";
import { addInputOnEdgeDrop } from "../flow/Nodes/BaseNodes/utils/addInputOnEdgeDrop";
import { inputNodes } from "../flow/Nodes/BaseNodes/utils/RegisterNodes";
import { getComputedNodes } from "../flow/Nodes/ComputeNodes/getComputedNodes";
import { $capabilities } from "../globalStore/capabilitiesStore";
import {
  $appState,
  $edges,
  $flowInst,
  $nodes,
  setEdges,
  setNodes,
} from "../globalStore/flowStore";
import { mapPrimitivesToNamespaced } from "../sync/namespaceUtils";
import { useSync } from "../sync/useSync";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { getColorFromEnum } from "../utils/colorUtils";

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
  const reactFlowInst = useReactFlow<CustomFlowNode, Edge>();
  const [opened, handlers] = useDisclosure(false);
  const appState = useStore($appState);
  const [pos, setPos] = useState<{ point: Point; nodes: CustomFlowNode[] }>({
    point: { x: 0, y: 0 },
    nodes: [],
  });

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

      return caps;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [caps]
  );

  const onNodesChange: OnNodesChange<CustomFlowNode> = useCallback(
    (changes: NodeChange<CustomFlowNode>[]) => {
      console.log(changes);
      setNodes(applyNodeChanges<CustomFlowNode>(changes, nodes));
    },
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
      <GroupContextMenu
        opened={opened}
        pos={pos}
        reactFlowInst={reactFlowInst}
      />
      <Stack
        h="100%"
        pos="absolute"
        justify="center"
        p="lg"
        opacity={appState.currentSubgraphId == 0 ? 0 : 1}
      >
        {Array(4)
          .fill(null)
          .map((_, index) => (
            <ColorSwatch
              key={index}
              size="24px"
              color={getColorFromEnum(index)[5]}
            />
          ))}
      </Stack>
      <Stack
        h="100%"
        pos="absolute"
        justify="center"
        right="0"
        p="lg"
        opacity={appState.currentSubgraphId == 0 ? 0 : 1}
      >
        {Array(3)
          .fill(null)
          .map((_, index) => (
            <ColorSwatch
              key={index}
              size="24px"
              color={getColorFromEnum(index)[5]}
            />
          ))}
      </Stack>
      <ReactFlow<CustomFlowNode, CustomFlowEdge>
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
        onClick={handlers.close}
        onSelectionContextMenu={(ev, nodes) => {
          ev.preventDefault();
          setPos({
            point: {
              x: ev.clientX,
              y: ev.clientY,
            },
            nodes: nodes,
          });
          handlers.open();
        }}
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </Box>
  );
};
