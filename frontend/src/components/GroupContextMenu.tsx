import { Paper, Menu, alpha } from "@mantine/core";
import { IconSettings } from "@tabler/icons-react";
import { theme } from "../theme";
import {
  CustomFlowNode,
  CustomGraphInstance,
} from "../flow/Nodes/CustomNodeType";
import { generateFlowId } from "../globalStore/flowStore";
// import { useStore } from "@nanostores/react";

export interface GroupContextMenuProps {
  opened: boolean;
  pos: {
    point: { x: number; y: number };
    nodes: CustomFlowNode[];
  };
  reactFlowInst: CustomGraphInstance;
}

export const GroupContextMenu = ({
  opened,
  pos,
  reactFlowInst,
}: GroupContextMenuProps) => {
  return "helo";
  // const appState = useStore($syncedAppState);

  const nodes = appState.main.nodes;
  return (
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
            const id = generateFlowId();
            const nodesNew = nodes.filter(
              (node) => !pos.nodes.map((n) => n.id).includes(node.id)
            );
            const groupNode: CustomFlowNode = {
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
              data: {
                capability: {
                  outputs: [],
                  inputs: [],
                  description: "Node for grouping nodes",
                  name: "Group",
                  namespace: "system",
                },
              },
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
      </Menu>
    </Paper>
  );
};
