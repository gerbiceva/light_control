import { NumberInput } from "@mantine/core";
import { Node, NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";

type ColorNodeData = { int: number };
type ColorNode = NodeProps<Node<ColorNodeData, "intPrimitive">>;

export const IntNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      type={"Color"}
      handle={
        <TypedHandle color={getColorFromEnum(BaseType.Int)[5]} id={"Int"} />
      }
      input={
        <NumberInput
          size="xs"
          defaultValue={data.int}
          className="nodrag"
          allowDecimal={false}
          onChange={(int) => {
            if (typeof int == "string") {
              return;
            }
            data.int = int;
          }}
          min={Number.MIN_SAFE_INTEGER}
          max={Number.MAX_SAFE_INTEGER}
        />
      }
    />
  );
});
