import {
  Box,
  ColorSwatch,
  Loader,
  LoadingOverlay,
  Stack,
  Title,
} from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useDisclosure } from "@mantine/hooks";
import { useStore } from "@nanostores/react";
import {
  Background,
  Controls,
  Edge,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type DefaultEdgeOptions,
  type FitViewOptions,
  type NodeTypes,
  type OnConnect,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Point } from "react-bezier-spline-editor/core";
import { GroupContextMenu } from "../components/GroupContextMenu";
import { isValidConnection } from "../flow/Edges/isValidCOnnection";
import { addInputOnEdgeDrop } from "../flow/Nodes/BaseNodes/utils/addInputOnEdgeDrop";
import { inputNodes } from "../flow/Nodes/BaseNodes/utils/RegisterNodes";
import { getComputedNodes } from "../flow/Nodes/ComputeNodes/getComputedNodes";
import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { $capabilities } from "../globalStore/capabilitiesStore";
import { $flowInst } from "../globalStore/flowStore";
import { mapPrimitivesToNamespaced } from "../sync/namespaceUtils";
import { getColorFromEnum } from "../utils/colorUtils";
import { addEdge, onEdgesChange, onNodesChange } from "../crdt/repo";
import { useYjsState } from "../crdt/globalSync";
import { useSubgraphs } from "../globalStore/subgraphStore";
import { getColoredEdge } from "../flow/Edges/getColoredEdge";

const fitViewOptions: FitViewOptions = {
  padding: 300,
};

const defaultEdgeOptions: DefaultEdgeOptions = {
  animated: true,
};

export const NodeView = () => {
  const caps = useStore($capabilities);
  const reactFlowInst = useReactFlow<CustomFlowNode, Edge>();
  const [opened, handlers] = useDisclosure(false);
  const { activeGraph } = useSubgraphs();
  const [pos, setPos] = useState<{ point: Point; nodes: CustomFlowNode[] }>({
    point: { x: 0, y: 0 },
    nodes: [],
  });
  const { nodes, edges, isReady } = useYjsState();

  useEffect(() => {
    reactFlowInst.fitView();
  }, [reactFlowInst]);

  useEffect(() => {
    $flowInst.set(reactFlowInst);
  }, [reactFlowInst]);

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

  const onConnect: OnConnect = useCallback((connection) => {
    console.log({ connection });
    addEdge(getColoredEdge(connection));
  }, []);

  if (!activeGraph || !isReady) {
    return (
      <LoadingOverlay
        visible={true}
        loaderProps={{
          children: (
            <Stack align="center" gap="xl">
              <Loader />
              <Title size="sm">Syncing</Title>
            </Stack>
          ),
        }}
      />
    );
  }

  return (
    <Box w="100%" h="100%" pos="relative">
      <GroupContextMenu
        opened={opened}
        pos={pos}
        reactFlowInst={reactFlowInst}
      />
      <Stack h="100%" pos="absolute" justify="center" p="lg">
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
      <Stack h="100%" pos="absolute" justify="center" right="0" p="lg">
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
        snapToGrid
        snapGrid={[50, 50]}
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
        <Background
          gap={50}
          size={3}
          offset={[100, 100]}
          style={{ opacity: 0.8 }}
        />
        <MiniMap />
      </ReactFlow>
    </Box>
  );
};
