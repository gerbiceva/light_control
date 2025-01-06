import { Avatar } from "@mantine/core";
import {
  SpotlightActionData,
  SpotlightActionGroupData,
} from "@mantine/spotlight";
import { useStore } from "@nanostores/react";
import { useCallback, useMemo } from "react";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { theme } from "../../theme";
import { generateNodeInstFromCapability } from "../../flow/Nodes/ComputeNodes/ComputeNodeFactory";
import { $nodes, setNodes } from "../../globalStore/flowStore";
import { Node } from "@xyflow/react";
import { inputNodesActions } from "../../flow/Nodes/BaseNodes/utils/SpotlightActions";

export const UseActions = () => {
  const capabilities = useStore($capabilities);
  const nodes = useStore($nodes);

  const addNode = useCallback(
    (node: Node) => {
      setNodes([...nodes, node]);
    },
    [nodes]
  );

  const actionsFromCapabilities: SpotlightActionData[] = useMemo(() => {
    return capabilities.map((cap) => {
      return {
        id: cap.id,
        label: cap.name,
        description: cap.description,
        onClick: () => {
          console.log({ cap });
          console.log(generateNodeInstFromCapability(cap));
          addNode(generateNodeInstFromCapability(cap));
        },
        leftSection: (
          <Avatar radius={0} color={theme.colors["gray"][5]}>
            {cap.name.slice(0, 2)}
          </Avatar>
        ),
        // description: (
        //   <Stack gap="8px">
        //     <Text size="xs">{cap.description}</Text>
        //     <Group wrap="nowrap" gap="3rem">
        //       <Group gap="xs">
        //         {cap.inputs.map((inp) => (
        //           <ColorSwatch
        //             key={inp.name}
        //             size="8px"
        //             color={getColorFromEnum(inp.type)[5]}
        //           />
        //         ))}
        //       </Group>
        //       <Group gap="xs">
        //         {cap.outputs.map((inp) => (
        //           <ColorSwatch
        //             key={inp.name}
        //             size="8px"
        //             color={getColorFromEnum(inp.type)[5]}
        //           />
        //         ))}
        //       </Group>
        //     </Group>
        //   </Stack>
        // ),
      };
    });
  }, [addNode, capabilities]);

  const actions: SpotlightActionGroupData[] = useMemo(() => {
    if (!theme.colors) {
      return [];
    }

    return [
      {
        group: "Inputs",
        actions: inputNodesActions,
      },
      { group: "Nodes", actions: [...actionsFromCapabilities] },
    ];
  }, [actionsFromCapabilities]);

  return actions;
};
