import { Avatar } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { useCallback, useMemo } from "react";
import { $serverCapabilities } from "../../globalStore/capabilitiesStore";
import { theme } from "../../theme";
import { generateNodeInstFromCapability } from "../../flow/Nodes/ComputeNodes/ComputeNodeFactory";
import { $nodes, setNodes } from "../../globalStore/flowStore";
import { Node } from "@xyflow/react";
import { inputNodesActions } from "../../flow/Nodes/BaseNodes/utils/SpotlightActions";

import { CustomSpotData } from "./CustomSpot/CustomSpotData";
import { CustomSpotlightGroups } from "./CustomSpot/CustomSpotlight";
import { getColorFromString } from "../../utils/colorUtils";

export const useActions = (): CustomSpotlightGroups[] => {
  const serverCapabilities = useStore($serverCapabilities);
  const nodes = useStore($nodes);

  const addNode = useCallback(
    (node: Node) => {
      setNodes([...nodes, node]);
    },
    [nodes]
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
