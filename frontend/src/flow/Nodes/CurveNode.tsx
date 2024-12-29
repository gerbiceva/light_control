import { memo, useState } from "react";
import { Node, NodeProps } from "@xyflow/react";
import { Text, Card, Group, Stack, useMantineTheme } from "@mantine/core";
import { TypedHandle } from "../Components/TypedHandle";
import { Point } from "react-bezier-spline-editor/core";
import { BezierSplineEditor } from "react-bezier-spline-editor/react";
import { BaseType } from "../../grpc/client_code/service";
import { getColorFromEnum } from "../../utils/colorUtils";

type CurveNodeData = { points: Point[] };
type CurveNode = NodeProps<Node<CurveNodeData, "curvePrimitive">>;

export const CurveNode = memo(({ data }: CurveNode) => {
  const theme = useMantineTheme();
  const [points, setPoints] = useState<Point[]>();

  return (
    <Card withBorder p="0">
      <Stack pb="0" gap="0">
        <Group bg="dark" p="xs">
          <Text c="white" size="xs">
            Curve
          </Text>
        </Group>

        <Group className="nodrag" p="xs" pr="0">
          <Card p="xl" withBorder radius="0">
            <BezierSplineEditor
              width={200}
              height={200}
              points={points}
              anchorPointProps={{
                color: theme.colors["dark"][7],
                fill: theme.colors["dark"][4],
              }}
              onPointsChange={(points) => {
                setPoints(points);
                data.points = points;
              }}
            />
          </Card>

          <TypedHandle color={getColorFromEnum(BaseType.Curve)[5]} id={"a"} />
        </Group>
      </Stack>
    </Card>
  );
});
