import { BaseType, NodeCapability } from "../../../grpc/client_code/service";

export const testingCapabilityNode: NodeCapability = {
  name: "Special multi mega node",
  description: "special node can do special things and is special",
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
    description: "zan le jto",
    inputs: [
      {
        name: "waaw",
        type: BaseType.Curve,
      },
    ],
    outputs: [],
  },
];
