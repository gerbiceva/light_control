import { memo } from "react";
import { Node, NodeProps } from "@xyflow/react";
import { Text, Card, Group, Stack, NumberInput } from "@mantine/core";
import { TypedHandle } from "../TypedHandle";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";

type ColorNodeData = { int: number };
type ColorNode = NodeProps<Node<ColorNodeData, "intPrimitive">>;

export const IntNode = memo(({ data }: ColorNode) => {
  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
            Integer
          </Text>
        </Group>

        <Group p="xs" pr="0">
          <NumberInput
            size="xs"
            defaultValue={data.int}
            className="nodrag"
            allowDecimal={false}
            onChange={(int) => {
              if (typeof int == "string") {
                return;
              }
              data.int = int;
            }}
            min={Number.MIN_SAFE_INTEGER}
            max={Number.MAX_SAFE_INTEGER}
          />
          <TypedHandle color={getColorFromEnum(BaseType.Int)[5]} id={"a"} />
        </Group>
      </Stack>
    </Card>
  );
});
