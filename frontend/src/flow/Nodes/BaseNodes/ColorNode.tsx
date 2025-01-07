import { memo } from "react";
import { Node, NodeProps } from "@xyflow/react";
import { ColorInput } from "@mantine/core";
import { TypedHandle } from "../TypedHandle";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { BaseType } from "../../../grpc/client_code/service";
import { BaseNodeElement } from "./BaseNodeElement";

type ColorNodeData = { color: string };
type ColorNode = NodeProps<Node<ColorNodeData, "Color">>;

export const ColorNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
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
            data.color = color;
          }}
        />
      }
    />
  );
});
