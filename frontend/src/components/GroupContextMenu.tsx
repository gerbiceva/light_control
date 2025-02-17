import { Paper, Menu, alpha } from "@mantine/core";
import { IconSettings } from "@tabler/icons-react";
import { $nodes, generateFlowId, setNodes } from "../globalStore/flowStore";
import { theme } from "../theme";
import { Node, ReactFlowInstance } from "@xyflow/react";
import { useStore } from "@nanostores/react";

export interface GroupContextMenuProps {
  opened: boolean;
  pos: {
    point: { x: number; y: number };
    nodes: Node[];
  };
  reactFlowInst: ReactFlowInstance;
}

export const GroupContextMenu = ({
  opened,
  pos,
  reactFlowInst,
}: GroupContextMenuProps) => {
  const nodes = useStore($nodes);
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
            // reactFlowInst.
            // console.log(pos.nodes[0].width, "helo");
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
      </Menu>
    </Paper>
  );
};
