import { NodeCapability } from "../../grpc/client_code/service";
import { Edge, Node, ReactFlowInstance } from "@xyflow/react";

export type CustomNodeData = {
  capability: NodeCapability;
  value?: string;
};
export type CustomFlowNode = Node<CustomNodeData>;
export type CustomFlowEdge = Edge;

export type InputNodeData = {
  value: unknown;
  readonly capability: NodeCapability;
};

export type CustomGraphInstance = ReactFlowInstance<
  CustomFlowNode,
  CustomFlowEdge
>;

export const isCustomFlowNode = (node: Node): node is CustomFlowNode => {
  return node.data.capability != undefined;
};

export const isCustomFlowEdge = (_edge: Edge): _edge is CustomFlowEdge => {
  return true;
};
