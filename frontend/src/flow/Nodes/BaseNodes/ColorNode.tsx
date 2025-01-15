import { memo } from "react";
import { NodeProps } from "@xyflow/react";
import { ColorInput } from "@mantine/core";
import { TypedHandle } from "../TypedHandle";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { BaseType } from "../../../grpc/client_code/service";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const ColorNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      namespace="inputs"
      type={"Color"}
      handle={
        <TypedHandle
          color={getColorFromEnum(BaseType.Color)["5"]}
          id={"Color"}
        />
      }
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
