import { memo } from "react";
import { NodeProps } from "@xyflow/react";
import { ColorInput } from "@mantine/core";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const ColorNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      namespace="inputs"
      type={"Color"}
      input={
        <ColorInput
          miw={"10rem"}
          size="xs"
          format="hsl"
          className="nodrag"
          onChange={(color) => {
            data.value = color;
          }}
        />
      }
    />
  );
});
