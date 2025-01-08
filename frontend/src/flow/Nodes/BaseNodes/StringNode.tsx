import { TextInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type StringNode = NodeProps<FlowNodeWithValue>;

export const StringNode = memo(({ data }: StringNode) => {
  return (
    <BaseNodeElement
      type={"Color"}
      handle={
        <TypedHandle
          color={getColorFromEnum(BaseType.String)[5]}
          id={"String"}
        />
      }
      input={
        <TextInput
          size="xs"
          defaultValue={data.value as string}
          className="nodrag"
          onChange={(ev) => {
            data.value = ev.target.value;
          }}
        />
      }
    />
  );
});
