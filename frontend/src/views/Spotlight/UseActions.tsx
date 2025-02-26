import { Avatar } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { useCallback, useMemo } from "react";
import { $serverCapabilities } from "../../globalStore/capabilitiesStore";
import { theme } from "../../theme";
import { generateNodeInstFromCapability } from "../../flow/Nodes/ComputeNodes/ComputeNodeFactory";
import { $nodes, setEdges, setNodes } from "../../globalStore/flowStore";
import { inputNodesActions } from "../../flow/Nodes/BaseNodes/utils/SpotlightActions";

import { CustomSpotData } from "./CustomSpot/CustomSpotData";
import { CustomSpotlightGroups } from "./CustomSpot/CustomSpotlight";
import { getColorFromString } from "../../utils/colorUtils";
import { $spotFilter } from "../../globalStore/spotlightFilterStore";
import { addColoredEdge } from "../../flow/Edges/addColoredEdge";
import { getCapFromNode } from "../../flow/Edges/typesFromConnection";
import { CustomFlowNode } from "../../flow/Nodes/CustomNodeType";

export const useActions = (): CustomSpotlightGroups[] => {
  const serverCapabilities = useStore($serverCapabilities);
  const nodes = useStore($nodes);

  const addNode = useCallback(
    (node: CustomFlowNode) => {
      const spotFilter = $spotFilter.get();
      const cap = getCapFromNode(node);
      setNodes([...nodes, node]);

      if (spotFilter && cap) {
        setEdges(
          spotFilter.type == "target"
            ? addColoredEdge({
                source: node.id,
                sourceHandle: cap.outputs.filter(
                  (port) => port.type == spotFilter.dataType,
                )[0].name,
                target: spotFilter.fromHandle.nodeId,
                targetHandle: spotFilter.fromHandle.id!,
              })
            : addColoredEdge({
                source: spotFilter.fromHandle.nodeId,
                sourceHandle: spotFilter.fromHandle.id!,
                target: node.id,
                targetHandle: cap.inputs.filter(
                  (port) => port.type == spotFilter.dataType,
                )[0].name,
              }),
        );
      }
    },
    [nodes],
  );

  // SERVER CAPABILITES ONLY
  const actionsFromCapabilities: CustomSpotData[] = useMemo(() => {
    return serverCapabilities.map((cap) => {
      return {
        id: cap.name,
        label: cap.name,
        description: cap.description,
        capability: cap,

        onClick: () => {
          addNode(generateNodeInstFromCapability(cap));
        },
        leftSection: (
          <Avatar radius={0} color={getColorFromString(cap.namespace)[5]}>
            {cap.name.slice(0, 2)}
          </Avatar>
        ),
      };
    });
  }, [addNode, serverCapabilities]);

  const actions: CustomSpotlightGroups[] = useMemo(() => {
    if (!theme.colors) {
      return [];
    }

    return [
      {
        groupName: "Primitive",
        data: inputNodesActions,
      },
      {
        groupName: "Nodes",
        data: actionsFromCapabilities,
      },
    ];
  }, [actionsFromCapabilities]);

  return actions;
};
