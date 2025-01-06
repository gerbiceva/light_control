import { Avatar } from "@mantine/core";
import { BaseType } from "../../../../grpc/client_code/service";
import { getColorFromEnum } from "../../../../utils/colorUtils";
import { Node } from "@xyflow/react";
import { addNode, generateFlowId } from "../../../../globalStore/flowStore";

export const inputNodesActions = [
  {
    id: "Int",
    label: "Int",
    description: "Whole number input",
    onClick: () => addNode(generateNodeInstFromInput("Int")),
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(BaseType.Int)[5]}>
        1
      </Avatar>
    ),
  },
  {
    id: "Float",
    label: "Float",
    description: "Floating point / decimal number",
    onClick: () => addNode(generateNodeInstFromInput("Float")),
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(BaseType.Float)[5]}>
        1.0
      </Avatar>
    ),
  },
  {
    id: "curve",
    label: "Curve",
    description: "Parametric curve input",
    onClick: () => addNode(generateNodeInstFromInput("curve")),
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(BaseType.Curve)[5]}>
        COL
      </Avatar>
    ),
  },
  {
    id: "Color",
    label: "Color",
    description: "HSV color input",
    onClick: () => addNode(generateNodeInstFromInput("Color")),
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(BaseType.Color)[5]}>
        HSV
      </Avatar>
    ),
  },
  {
    id: "String",
    label: "String",
    description: "String input",
    onClick: () => addNode(generateNodeInstFromInput("String")),
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(BaseType.String)[5]}>
        STR
      </Avatar>
    ),
  },
];

export const generateNodeInstFromInput = (
  type: string
  // inst: ReactFlowInstance<FlowNode, Edge>
): Node => {
  return {
    id: generateFlowId(),
    type: type,
    position: {
      x: 0,
      y: 0,
    },
    // position: inst.screenToFlowPosition({
    //   x: window.clientX,
    //   y: clientY,
    // }),
    data: {},
  };
};
