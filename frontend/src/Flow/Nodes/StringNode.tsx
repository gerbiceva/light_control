import { memo } from "react";
import { Node, NodeProps } from "@xyflow/react";
import {
  Text,
  Card,
  Group,
  Stack,
  useMantineTheme,
  TextInput,
} from "@mantine/core";
import { TypedHandle } from "../Components/TypedHandle";

type StringNodeData = { str: string };
type StringNode = NodeProps<Node<StringNodeData, "stringPrimitive">>;

export const StringNode = memo(({ data }: StringNode) => {
  const theme = useMantineTheme();

  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
            String
          </Text>
        </Group>

        <Group pl="md">
          <TextInput
            size="xs"
            defaultValue={data.str}
            className="nodrag"
            onChange={(ev) => {
              data.str = ev.target.value;
            }}
          />
          <TypedHandle color={theme.colors["pink"][5]} id={"a"} />
        </Group>
      </Stack>
    </Card>
  );
});
