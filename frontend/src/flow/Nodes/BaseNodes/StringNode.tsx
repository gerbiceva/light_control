import { TextInput } from "@mantine/core";
import { Node, NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";

type StringNodeData = { str: string };
type StringNode = NodeProps<Node<StringNodeData, "stringPrimitive">>;

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
          defaultValue={data.str}
          className="nodrag"
          onChange={(ev) => {
            data.str = ev.target.value;
          }}
        />
      }
    />
  );
});
