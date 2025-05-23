import { BaseType, NodeCapability } from "../../../grpc/client_code/service";

export const testingCapabilityNode: NodeCapability = {
  name: "Special multi mega node",
  description: "special node can do special things and is special",
  namespace: "testing func",
  inputs: [
    {
      name: "Color for led",
      type: BaseType.Color,
    },
    {
      name: "Pederchina",
      type: BaseType.Float,
    },
    {
      name: "VDIM",
      type: BaseType.Int,
    },
    {
      name: "Name",
      type: BaseType.String,
    },
  ],
  outputs: [
    {
      name: "VDIM",
      type: BaseType.Int,
    },
    {
      name: "Curve",
      type: BaseType.Curve,
    },
  ],
};

export const testCapabilitiesList: NodeCapability[] = [
  testingCapabilityNode,
  {
    name: "nekaj",
    namespace: "math",
    description: "zan le jto",
    inputs: [
      {
        name: "waaw",
        type: BaseType.Curve,
      },
    ],
    outputs: [],
  },
  {
    name: "nekaj2",
    namespace: "math",
    description: "zan le jto",
    inputs: [
      {
        name: "waaw",
        type: BaseType.Color,
      },
    ],
    outputs: [],
  },
  {
    name: "nekaj3",
    namespace: "math",
    description: "zan le jto",
    inputs: [
      {
        name: "waaw",
        type: BaseType.Int,
      },
    ],
    outputs: [],
  },
  {
    name: "nekaj4",
    description: "zan le jto",
    namespace: "things",
    inputs: [
      {
        name: "waaw",
        type: BaseType.Int,
      },
    ],
    outputs: [],
  },
];
