import { Node } from "../../../grpc/client_code/service";

import { Card, Group, SimpleGrid, Stack, Text, Tooltip } from "@mantine/core";
import { DynamicPort } from "./DynamicPort";
import { theme } from "../../../theme";
import { NodeProps, Node as FlowNode } from "@xyflow/react";

export const GenerateComputeNodeFromCapability = (capability: Node) => {
  const inputStack = (
    <Stack gap="xs">
      {capability.inputs.map((input) => (
        <DynamicPort port={input} type="target" key={input.name} />
      ))}
    </Stack>
  );

  const outputStack = (
    <Stack gap="xs">
      {capability.outputs.map((input) => (
        <DynamicPort port={input} type="source" key={input.name} />
      ))}
    </Stack>
  );

  const ComputeNode = ({ selected }: NodeProps<FlowNode>) => {
    return (
      <Card
        withBorder
        p="0"
        style={{
          borderColor: selected ? theme.colors["dark"][5] : undefined,
          border: "2px solid",
        }}
      >
        <Stack pb="0" gap="0">
          <Group bg="dark" p="xs" w="100%" justify="space-between">
            <Text c="white" size="xs">
              {capability.name}
            </Text>
            <Tooltip label={capability.description}>
              <Text c="white" fw="bold">
                ?
              </Text>
            </Tooltip>
          </Group>

          <SimpleGrid cols={2} py="xs">
            {inputStack}
            {outputStack}
          </SimpleGrid>
        </Stack>
      </Card>
    );
  };

  return ComputeNode;
};
