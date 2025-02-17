import { alpha, Box, Menu, Paper, useMantineTheme } from "@mantine/core";

import "@xyflow/react/dist/style.css";

import { useCallback, useEffect, useMemo, useState } from "react";
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
  Node,
} from "@xyflow/react";
import { getComputedNodes } from "../flow/Nodes/ComputeNodes/getComputedNodes";
import {
  $edges,
  $flowInst,
  $nodes,
  generateFlowId,
  setEdges,
  setNodes,
  addNode,
} from "../globalStore/flowStore";
import { useStore } from "@nanostores/react";
import { inputNodes } from "../flow/Nodes/BaseNodes/utils/RegisterNodes";
import { useSync } from "../sync/useSync";
import { addColoredEdge } from "../flow/Edges/addColoredEdge";
import { isValidConnection } from "../flow/Edges/isValidCOnnection";
import { $capabilities } from "../globalStore/capabilitiesStore";
import { addInputOnEdgeDrop } from "../flow/Nodes/BaseNodes/utils/addInputOnEdgeDrop";
import { mapPrimitivesToNamespaced } from "../sync/namespaceUtils";
import {
  IconMessageCircle,
  IconPhoto,
  IconSettings,
} from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { Point } from "react-bezier-spline-editor/core";

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
  const [opened, handlers] = useDisclosure(false);
  const theme = useMantineTheme();
  const [pos, setPos] = useState<{ point: Point; nodes: Node[] }>({
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
    [caps],
  );

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes(applyNodeChanges(changes, nodes)),
    [nodes],
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges(applyEdgeChanges(changes, edges)),
    [edges],
  );

  const onConnect: OnConnect = useCallback((connection) => {
    setEdges(addColoredEdge(connection));
  }, []);

  return (
    <Box w="100%" h="100%" pos="relative">
      <Paper
        display={opened ? "block" : "none"}
        p="sm"
        withBorder
        pos="absolute"
        style={{
          top: pos.point.y - 100,
          left: pos.point.x + 10,
          zIndex: 10,
        }}
      >
        <Menu>
          <Menu.Label>Group menu</Menu.Label>
          <Menu.Item
            leftSection={<IconSettings size={14} />}
            onClick={() => {
              const rect = reactFlowInst.getNodesBounds(pos.nodes);
              // reactFlowInst.
              console.log(pos.nodes[0].width, "helo");
              const id = generateFlowId();
              const nodesNew = nodes.filter(
                (node) => !pos.nodes.map((n) => n.id).includes(node.id),
              );
              const groupNode: Node = {
                id,
                type: "group",
                position: { x: rect.x, y: rect.y },
                style: {
                  width: rect.width + 30,
                  height: rect.height + 30,
                  padding: "2rem",
                  border: "2px solid gray",
                  borderRadius: theme.defaultRadius,
                  borderColor: theme.colors.cyan[3],
                  backgroundColor: alpha(theme.colors.cyan[1], 0.2),
                },
                data: {},
              };

              const updatedNodes = pos.nodes.map((n) => {
                return {
                  ...n,
                  extent: "parent",
                  parentId: id,
                  position: {
                    x: n.position.x - rect.x, // Translate to parent's X
                    y: n.position.y - rect.y, // Translate to parent's Y
                  },
                };
              });

              //@ts-expect-error different node types
              setNodes([...nodesNew, groupNode, ...updatedNodes]);
            }}
          >
            Group selection <kbd>shift + G</kbd>
          </Menu.Item>
          <Menu.Item leftSection={<IconMessageCircle size={14} />}>
            Messages
          </Menu.Item>
          <Menu.Item leftSection={<IconPhoto size={14} />}>Gallery</Menu.Item>
        </Menu>
      </Paper>
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
        onClick={handlers.close}
        onNodeContextMenu={() => {
          console.log("PANEL");
        }}
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
