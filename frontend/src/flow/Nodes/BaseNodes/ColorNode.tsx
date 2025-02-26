import { memo } from "react";
import { ColorInput } from "@mantine/core";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";
import { NodeProps } from "@xyflow/react";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const ColorNode = memo(({ selected, data }: ColorNode) => {
  return (
    <BaseNodeElement
      selected={selected}
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
