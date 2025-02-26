import { Card, useMantineTheme } from "@mantine/core";
import { NodeProps } from "@xyflow/react";
import { memo, useState } from "react";
import { Point } from "react-bezier-spline-editor/core";
import { BezierSplineEditor } from "react-bezier-spline-editor/react";
import { BaseNodeElement } from "./BaseNodeElement";
import { FlowNodeWithValue } from "./utils/inputNodeType";

type CurveNode = NodeProps<FlowNodeWithValue>;

/**
 * @deprecated, curves area created from string
 */
export const CurveNode = memo(({ data, selected }: CurveNode) => {
  const theme = useMantineTheme();
  const [points, setPoints] = useState<Point[]>();

  return (
    <BaseNodeElement
      selected={selected}
      namespace="inputs"
      type={"Curve"}
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
