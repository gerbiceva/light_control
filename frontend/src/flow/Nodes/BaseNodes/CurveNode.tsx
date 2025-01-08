import { Card, useMantineTheme } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo, useState } from "react";
import { Point } from "react-bezier-spline-editor/core";
import { BezierSplineEditor } from "react-bezier-spline-editor/react";
import { BaseType } from "../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../utils/colorUtils";
import { TypedHandle } from "../TypedHandle";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type CurveNode = NodeProps<FlowNodeWithValue>;

export const CurveNode = memo(({ data }: CurveNode) => {
  const theme = useMantineTheme();
  const [points, setPoints] = useState<Point[]>();

  return (
    <BaseNodeElement
      type={"Curve"}
      handle={
        <TypedHandle color={getColorFromEnum(BaseType.Curve)[5]} id={"Curve"} />
      }
      input={
        <Card p="xl" withBorder radius="0" className="nodrag">
          <BezierSplineEditor
            width={200}
            height={200}
            points={points}
            displayRelativePoints
            anchorPointProps={{
              color: theme.colors["dark"][7],
              fill: theme.colors["dark"][4],
            }}
            onPointsChange={(points) => {
              setPoints(points);
              data.value = points;
            }}
          />
        </Card>
      }
    />
  );
});
