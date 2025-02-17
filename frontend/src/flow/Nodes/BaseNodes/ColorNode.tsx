import { memo } from "react";
import { ColorInput } from "@mantine/core";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = FlowNodeWithValue;

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
