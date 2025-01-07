import { BaseType, NodeCapability } from "../../../../grpc/client_code/service";

// interface NodeCapabilityWithJSX extends NodeCapability {
// }

export const baseCapabilities: NodeCapability[] = [
  {
    name: "Int",
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
    description: "String input",
  },
];
