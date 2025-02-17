import { NumberInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const IntNode = memo(({ data, selected }: ColorNode) => {
  return (
    <BaseNodeElement
      selected={selected}
      namespace="inputs"
      type={"Int"}
      input={
        <NumberInput
          size="xs"
          defaultValue={data.value as number}
          className="nodrag"
          allowDecimal={false}
          onChange={(int) => {
            if (typeof int == "string") {
              return;
            }
            data.value = int;
          }}
          min={Number.MIN_SAFE_INTEGER}
          max={Number.MAX_SAFE_INTEGER}
        />
      }
    />
  );
});
