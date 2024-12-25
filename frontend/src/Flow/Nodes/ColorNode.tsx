import { memo } from "react";
import { Node, NodeProps } from "@xyflow/react";
import {
  Text,
  Card,
  Group,
  Stack,
  ColorInput,
  useMantineTheme,
} from "@mantine/core";
import { TypedHandle } from "../Components/TypedHandle";

type ColorNodeData = { color: string };
type ColorNode = NodeProps<Node<ColorNodeData, "colorPrimitive">>;

export const ColorNode = memo(({ data }: ColorNode) => {
  const theme = useMantineTheme();

  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
            Color
          </Text>
        </Group>

        <Group pl="md">
          <ColorInput
            size="xs"
            defaultValue={data.color}
            format="hsl"
            className="nodrag"
            onChange={(color) => {
              data.color = color;
            }}
          />
          <TypedHandle color={theme.colors["cyan"][5]} id={"a"} />
        </Group>
      </Stack>
    </Card>
  );
});
