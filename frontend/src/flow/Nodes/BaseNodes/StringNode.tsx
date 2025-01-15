import { TextInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type StringNode = NodeProps<FlowNodeWithValue>;

export const StringNode = memo(({ data }: StringNode) => {
  return (
    <BaseNodeElement
      namespace="inputs"
      type={"String"}
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
