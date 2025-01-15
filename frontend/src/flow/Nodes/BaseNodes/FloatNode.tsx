import { NumberInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const FloatNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      type={"Float"}
      handle={
        <TypedHandle color={getColorFromEnum(BaseType.Float)[5]} id={"Float"} />
      }
      input={
        <NumberInput
          size="xs"
          defaultValue={data.value as number}
          className="nodrag"
          allowDecimal={true}
          decimalScale={2}
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
