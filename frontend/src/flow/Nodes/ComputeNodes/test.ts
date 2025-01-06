import { BaseType, Node } from "../../../grpc/client_code/service";

export const testingCapabilityNode: Node = {
  name: "Special multi mega node",
  description: "special node can do special things and is special",
  id: "12",
  inputs: [
    {
      description: "changes led colors",
      name: "Color for led",
      type: BaseType.Color,
    },
    {
      description: "helo melo belo",
      name: "Pederchina",
      type: BaseType.Float,
    },
    {
      description: "controlls intensity",
      name: "VDIM",
      type: BaseType.Int,
    },
    {
      description: "gives it a name for no reason whatsoever",
      name: "Name",
      type: BaseType.String,
    },
  ],
  outputs: [
    {
      description: "controlls nekaj",
      name: "VDIM",
      type: BaseType.Int,
    },
    {
      description: "gives it a name for no reason whatsoever",
      name: "Curve",
      type: BaseType.Curve,
    },
  ],
};

export const testCapabilitiesList: Node[] = [testingCapabilityNode];
