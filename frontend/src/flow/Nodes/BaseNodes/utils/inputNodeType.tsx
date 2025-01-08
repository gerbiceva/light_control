import { Node as FlowNode } from "@xyflow/react";

export type FlowNodeWithValue = FlowNode<{ value: unknown }>;

export const isFlowNodeWithValue = (
  node: FlowNode
): node is FlowNodeWithValue => {
  return (node as FlowNodeWithValue).data.value != undefined;
};
