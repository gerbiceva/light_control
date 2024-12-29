import { Flex, Text, Tooltip } from "@mantine/core";
import { Port } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "./nodeUtils";
import { TypedHandle } from "../../Components/TypedHandle";
import { HandleType } from "@xyflow/react";

export interface IPortProps {
  port: Port;
  type: HandleType;
}

export const DynamicPort = ({ port, type }: IPortProps) => {
  const col = getColorFromEnum(port.type);
  return (
    <Tooltip label={port.description}>
      <Flex
        direction={type == "source" ? "row-reverse" : "row"}
        align="center"
        gap="sm"
      >
        <TypedHandle color={col[5]} id={port.name} type={type} />
        <Text size="xs">{port.name}</Text>
      </Flex>
    </Tooltip>
  );
};
