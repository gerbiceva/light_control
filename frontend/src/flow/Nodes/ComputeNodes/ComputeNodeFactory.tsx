import { NodeCapability } from "../../../grpc/client_code/service";

import { Card, Group, SimpleGrid, Stack, Text, Tooltip } from "@mantine/core";
import { DynamicPort } from "./DynamicPort";

import { NodeProps, Node as FlowNode } from "@xyflow/react";
import { $flowInst, generateFlowId } from "../../../globalStore/flowStore";
import { $frozenMousePos } from "../../../globalStore/mouseStore";
import { getColorFromString } from "../../../utils/colorUtils";
import { mergeNamespaceAndType } from "../../../sync/namespaceUtils";

export const generateNodeInstFromCapability = (
  capability: NodeCapability
): FlowNode => {
  const pos = $flowInst.get()?.screenToFlowPosition($frozenMousePos.get());
  return {
    id: generateFlowId(),
    type: mergeNamespaceAndType(capability.namespace, capability.name),
    position: pos || {
      x: 0,
      y: 0,
    },
    data: {},
  };
};

export const generateComputeNodeFromCapability = (
  capability: NodeCapability
) => {
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

  const ComputeNode = ({ selected, id }: NodeProps<FlowNode>) => {
    return (
      <Card withBorder p="0" shadow={selected ? "lg" : undefined}>
        <Stack pb="0" gap="0">
          <Group
            bg="dark"
            p="xs"
            w="100%"
            justify="space-between"
            style={{
              borderBottom: `4px solid ${
                getColorFromString(capability.namespace)[5]
              }`,
            }}
          >
            <Text
              c={getColorFromString(capability.namespace)[2]}
              size="xs"
              fw="bold"
              maw={"200px"}
            >
              {capability.name}
            </Text>
            {/* TODO: remove node ids */}
            <Tooltip label={capability.description + " : " + id}>
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
