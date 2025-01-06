import { BaseType, Node } from "../../../../grpc/client_code/service";

export const baseCapabilities: Node[] = [
  {
    id: "Int",
    name: "Int",
    description: "Whole number input",
    inputs: [],
    outputs: [
      {
        description: "Whole number output",
        name: "Int",
        type: BaseType.Int,
      },
    ],
  },
  {
    id: "Float",
    name: "Float",
    inputs: [],
    outputs: [
      {
        name: "Float",
        description: "Floating point / decimal number",
        type: BaseType.Float,
      },
    ],
    description: "Floating point / decimal number",
  },
  {
    id: "Curve",
    name: "Curve",
    inputs: [],
    outputs: [
      {
        name: "Curve",
        description: "Parametric curve input",
        type: BaseType.Curve,
      },
    ],
    description: "Parametric curve input",
  },
  {
    id: "Color",
    name: "Color",
    inputs: [],
    outputs: [
      {
        name: "Color",
        description: "HSV color input",
        type: BaseType.Color,
      },
    ],
    description: "HSV color input",
  },
  {
    id: "String",
    name: "String",
    inputs: [],
    outputs: [
      {
        name: "String",
        description: "String",
        type: BaseType.String,
      },
    ],
    description: "String input",
  },
];
