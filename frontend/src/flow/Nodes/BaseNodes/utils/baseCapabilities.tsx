import { BaseType, NodeCapability } from "../../../../grpc/client_code/service";

export const primitiveCapabilities: NodeCapability[] = [
  {
    name: "Int",
    namespace: "primitive",
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
    namespace: "primitive",
    description: "Floating point / decimal number",
  },
  // {
  //   name: "Curve",
  //   inputs: [],
  //   outputs: [
  //     {
  //       name: "Curve",
  //       type: BaseType.Curve,
  //     },
  //   ],
  //   namespace: "primitive",
  //   description: "Parametric curve input",
  // },
  {
    name: "Color",
    inputs: [],
    outputs: [
      {
        name: "Color",
        type: BaseType.Color,
      },
    ],
    namespace: "primitive",
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
    namespace: "primitive",
    description: "String input",
  },
];

export const getBaseCapabilityFromType = (
  type: string,
): NodeCapability | undefined => {
  return primitiveCapabilities.find((capability) => capability.name === type);
};
