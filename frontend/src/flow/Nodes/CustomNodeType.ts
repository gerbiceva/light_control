import { NodeCapability } from "../../grpc/client_code/service";
import { Node } from "@xyflow/react";

export type CustomNodeData = {
  capability: NodeCapability;
};
export type CustomFlowNode = Node<CustomNodeData>;

export type InputNodeData = {
  value: unknown;
  capability: NodeCapability;
};
