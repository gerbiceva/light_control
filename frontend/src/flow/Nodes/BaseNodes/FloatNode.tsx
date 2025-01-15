import { NumberInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const FloatNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      namespace="inputs"
      type={"Float"}
      input={
        <NumberInput
          size="xs"
          defaultValue={data.value as number}
          className="nodrag"
          allowDecimal={true}
          decimalScale={2}
          step={0.1}
          fixedDecimalScale
          onChange={(int) => {
            if (typeof int == "string") {
              return;
            }
            data.value = int;
          }}
        />
      }
    />
  );
});
