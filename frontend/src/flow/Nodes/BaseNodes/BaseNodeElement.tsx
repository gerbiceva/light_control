import { Card, Stack, Group, Text, Tooltip } from "@mantine/core";

interface BaseNodeElementProps {
  type: string;
  handle: JSX.Element;
  input: JSX.Element;
}

export const BaseNodeElement = ({
  handle,
  input,
  type,
}: BaseNodeElementProps) => {
  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
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
