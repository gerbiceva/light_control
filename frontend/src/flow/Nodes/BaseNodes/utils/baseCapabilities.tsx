import { BaseType, NodeCapability } from "../../../../grpc/client_code/service";

export const baseCapabilities: NodeCapability[] = [
  {
    name: "Int",
    namespace: "inputs",
    description: "Whole number input",
    inputs: [],
    outputs: [
      {
        name: "Int",
        type: BaseType.Int,
      },
    ],
  },
  {
    name: "Float",
    inputs: [],
    outputs: [
      {
        name: "Float",
        type: BaseType.Float,
      },
    ],
    namespace: "inputs",
    description: "Floating point / decimal number",
  },
  {
    name: "Curve",
    inputs: [],
    outputs: [
      {
        name: "Curve",
        type: BaseType.Curve,
      },
    ],
    namespace: "inputs",
    description: "Parametric curve input",
  },
  {
    name: "Color",
    inputs: [],
    outputs: [
      {
        name: "Color",
        type: BaseType.Color,
      },
    ],
    namespace: "inputs",
    description: "HSV color input",
  },
  {
    name: "String",
    inputs: [],
    outputs: [
      {
        name: "String",
        type: BaseType.String,
      },
    ],
    namespace: "inputs",
    description: "String input",
  },
];
