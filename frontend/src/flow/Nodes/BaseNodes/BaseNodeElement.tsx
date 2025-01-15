import { Card, Stack, Group, Text, Tooltip } from "@mantine/core";
import { getColorFromString } from "../../../utils/colorUtils";

interface BaseNodeElementProps {
  type: string;
  handle: JSX.Element;
  input: JSX.Element;
  namespace: string;
}

export const BaseNodeElement = ({
  handle,
  input,
  type,
  namespace,
}: BaseNodeElementProps) => {
  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group
          bg="dark"
          p="sm"
          style={{
            borderBottom: `4px solid ${getColorFromString(namespace)[5]}`,
          }}
        >
          <Text c={getColorFromString(namespace)[2]} size="xs" fw="bold">
            {type}
          </Text>
        </Group>

        <Tooltip label={type}>
          <Group p="xs" pr="0">
            {input}
            {handle}
          </Group>
        </Tooltip>
      </Stack>
    </Card>
  );
};
