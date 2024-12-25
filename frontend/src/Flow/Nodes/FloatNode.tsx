import { memo } from "react";
import { Node, NodeProps } from "@xyflow/react";
import {
  Text,
  Card,
  Group,
  Stack,
  useMantineTheme,
  NumberInput,
} from "@mantine/core";
import { TypedHandle } from "../Components/TypedHandle";

type ColorNodeData = { float: number };
type ColorNode = NodeProps<Node<ColorNodeData, "floatPrimitive">>;

export const FloatNode = memo(({ data }: ColorNode) => {
  const theme = useMantineTheme();

  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
            Float
          </Text>
        </Group>

        <Group pl="md">
          <NumberInput
            size="xs"
            defaultValue={data.float}
            className="nodrag"
            allowDecimal={true}
            decimalScale={2}
            fixedDecimalScale
            onChange={(int) => {
              if (typeof int == "string") {
                return;
              }
              data.float = int;
            }}
          />
          <TypedHandle color={theme.colors["green"][5]} id={"a"} />
        </Group>
      </Stack>
    </Card>
  );
});
