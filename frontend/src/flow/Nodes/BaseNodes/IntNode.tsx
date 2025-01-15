import { NumberInput } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo } from "react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type ColorNode = NodeProps<FlowNodeWithValue>;

export const IntNode = memo(({ data }: ColorNode) => {
  return (
    <BaseNodeElement
      type={"Int"}
      handle={
        <TypedHandle color={getColorFromEnum(BaseType.Int)[5]} id={"Int"} />
      }
      input={
        <NumberInput
          size="xs"
          defaultValue={data.value as number}
          className="nodrag"
          allowDecimal={false}
          onChange={(int) => {
            if (typeof int == "string") {
              return;
            }
            data.value = int;
          }}
          min={Number.MIN_SAFE_INTEGER}
          max={Number.MAX_SAFE_INTEGER}
        />
      }
    />
  );
});
