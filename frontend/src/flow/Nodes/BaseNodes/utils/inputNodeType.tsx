import { Node as FlowNode } from "@xyflow/react";
import { InputNodeData } from "../../CustomNodeType";

export type FlowNodeWithValue = FlowNode<InputNodeData, "primitiveNode">;

export const isFlowNodeWithValue = (
  node: FlowNode,
): node is FlowNodeWithValue => {
  return (node as FlowNodeWithValue).data.value != undefined;
};
