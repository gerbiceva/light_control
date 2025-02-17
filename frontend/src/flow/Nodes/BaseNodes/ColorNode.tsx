import { memo } from "react";
import { ColorInput } from "@mantine/core";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";
import { NodeProps } from "@xyflow/react";

type ColorNode = FlowNodeWithValue;

export const ColorNode = memo(({ selected }: ColorNode) => {
  return (
    <BaseNodeElement
      selected={selected?.data.value}
      namespace="inputs"
      type={"Color"}
      input={
        <ColorInput
          miw={"10rem"}
          size="xs"
          format="hsl"
          className="nodrag"
          onChange={(color) => {
            selected.data.value = color;
          }}
        />
      }
    />
  );
});
