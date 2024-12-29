import { Avatar } from "@mantine/core";
import {
  SpotlightActionData,
  SpotlightActionGroupData,
} from "@mantine/spotlight";
import { useStore } from "@nanostores/react";
import { useMemo } from "react";
import { getColorFromEnum } from "../../utils/colorUtils";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { BaseType } from "../../grpc/client_code/service";
import { theme } from "../../theme";

export const UseActions = () => {
  const capabilities = useStore($capabilities);

  const actionsFromCapabilities: SpotlightActionData[] = useMemo(() => {
    return capabilities.map((cap) => ({
      id: cap.id,
      label: cap.name,
      description: cap.description,

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
      onClick: () => console.log("Home"),
      leftSection: (
        <Avatar radius={0} color={theme.colors["gray"][5]}>
          {cap.name.slice(0, 2)}
        </Avatar>
      ),
    }));
  }, [capabilities]);

  const actions: SpotlightActionGroupData[] = useMemo(() => {
    if (!theme.colors) {
      return [];
    }

    return [
      {
        group: "Inputs",
        actions: [
          {
            id: "Int",
            label: "Int",
            description: "Whole number input",
            onClick: () => console.log("Home"),
            leftSection: (
              <Avatar radius={0} color={getColorFromEnum(BaseType.Int)[5]}>
                1
              </Avatar>
            ),
          },
          {
            id: "Float",
            label: "Float",
            description: "Floating point / decimal number",
            onClick: () => console.log("Home"),
            leftSection: (
              <Avatar radius={0} color={getColorFromEnum(BaseType.Float)[5]}>
                1.0
              </Avatar>
            ),
          },
          {
            id: "curve",
            label: "Curve",
            description: "Parametric curve input",
            onClick: () => console.log("Dashboard"),
            leftSection: (
              <Avatar radius={0} color={getColorFromEnum(BaseType.Curve)[5]}>
                COL
              </Avatar>
            ),
          },
          {
            id: "Color",
            label: "Color",
            description: "HSV color input",
            onClick: () => console.log("Dashboard"),
            leftSection: (
              <Avatar radius={0} color={getColorFromEnum(BaseType.Color)[5]}>
                HSV
              </Avatar>
            ),
          },
          {
            id: "String",
            label: "String",
            description: "String input",
            onClick: () => console.log("Dashboard"),
            leftSection: (
              <Avatar radius={0} color={getColorFromEnum(BaseType.String)[5]}>
                STR
              </Avatar>
            ),
          },
        ],
      },
      { group: "Nodes", actions: [...actionsFromCapabilities] },
    ];
  }, [theme.colors, actionsFromCapabilities]);

  return actions;
};
