import { Card, Stack, Group, Text, Tooltip } from "@mantine/core";
import {
  getColorFromEnum,
  getColorFromString,
} from "../../../utils/colorUtils";
import { BaseType } from "../../../grpc/client_code/service";
import { TypedHandle } from "../TypedHandle";

interface BaseNodeElementProps {
  type: string;
  input: JSX.Element;
  namespace: string;
}

export const BaseNodeElement = ({
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
            <TypedHandle
              //@ts-expect-error type will be of type basetype
              color={getColorFromEnum(BaseType[type])["5"]}
              id={type}
            />
          </Group>
        </Tooltip>
      </Stack>
    </Card>
  );
};
