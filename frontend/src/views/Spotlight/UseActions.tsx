import { Avatar } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { useCallback, useMemo } from "react";
import { $serverCapabilities } from "../../globalStore/capabilitiesStore";
import { theme } from "../../theme";
import { generateNodeInstFromCapability } from "../../flow/Nodes/ComputeNodes/ComputeNodeFactory";
import { inputNodesActions } from "../../flow/Nodes/BaseNodes/utils/SpotlightActions";

import { CustomSpotData } from "./CustomSpot/CustomSpotData";
import { CustomSpotlightGroups } from "./CustomSpot/CustomSpotlight";
import { getColorFromString } from "../../utils/colorUtils";
import { $spotFilter } from "../../globalStore/spotlightFilterStore";
import { getColoredEdge } from "../../flow/Edges/getColoredEdge";
import { CustomFlowNode } from "../../flow/Nodes/CustomNodeType";
import { addEdge, addNode } from "../../crdt/repo";

export const useActions = (): CustomSpotlightGroups[] => {
  const serverCapabilities = useStore($serverCapabilities);

  const createNewNode = useCallback((node: CustomFlowNode) => {
    const spotFilter = $spotFilter.get();
    const cap = node.data.capability;
    addNode(node);

    if (spotFilter && cap) {
      if (spotFilter.type == "target") {
        cap.outputs.forEach((port, i) => {
          if (port.type == spotFilter.dataType) {
            return addEdge(
              getColoredEdge({
                source: node.id,
                sourceHandle: i.toString(),
                target: spotFilter.fromHandle.nodeId,
                targetHandle: spotFilter.fromHandle.id!,
              })
            );
          }
        });
      } else {
        cap.outputs.forEach((port, i) => {
          if (port.type == spotFilter.dataType) {
            return addEdge(
              getColoredEdge({
                source: spotFilter.fromHandle.nodeId,
                sourceHandle: spotFilter.fromHandle.id!,
                target: node.id,
                targetHandle: i.toString(),
              })
            );
          }
        });
      }
    }
  }, []);

  // SERVER CAPABILITES ONLY
  const actionsFromCapabilities: CustomSpotData[] = useMemo(() => {
    return serverCapabilities.map((cap) => {
      return {
        id: cap.name,
        label: cap.name,
        description: cap.description,
        capability: cap,

        onClick: () => {
          createNewNode(generateNodeInstFromCapability(cap));
        },
        leftSection: (
          <Avatar radius={0} color={getColorFromString(cap.namespace)[5]}>
            {cap.name.slice(0, 2)}
          </Avatar>
        ),
      };
    });
  }, [createNewNode, serverCapabilities]);

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
