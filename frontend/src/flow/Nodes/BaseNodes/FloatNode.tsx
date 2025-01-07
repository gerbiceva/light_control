import { NumberInput } from "@mantine/core";
import { Node, NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";

type ColorNodeData = { float: number };
type ColorNode = NodeProps<Node<ColorNodeData, "floatPrimitive">>;

export const FloatNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      type={"Color"}
      handle={
        <TypedHandle color={getColorFromEnum(BaseType.Float)[5]} id={"Float"} />
      }
      input={
        <NumberInput
          size="xs"
          defaultValue={data.float}
          className="nodrag"
          allowDecimal={true}
          decimalScale={2}
          fixedDecimalScale
          onChange={(int) => {
            if (typeof int == "string") {
              return;
            }
            data.float = int;
          }}
        />
      }
    />
  );
});
