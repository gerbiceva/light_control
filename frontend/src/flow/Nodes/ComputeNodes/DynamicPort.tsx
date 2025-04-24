import { Flex, Text } from "@mantine/core";
import { Port } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { HandleType } from "@xyflow/react";

export interface IPortProps {
  port: Port;
  type: HandleType;
  index: number;
}

export const DynamicPort = ({ port, type, index }: IPortProps) => {
  const col = getColorFromEnum(port.type);
  return (
    <Flex
      direction={type == "source" ? "row-reverse" : "row"}
      align="center"
      gap="sm"
    >
      <TypedHandle color={col[5]} id={index.toString()} type={type} />
      <Text size="xs">{port.name}</Text>
    </Flex>
  );
};
