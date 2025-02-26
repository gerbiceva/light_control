import { Node as FlowNode } from "@xyflow/react";

type NodeData = {
  value: unknown;
};
export type FlowNodeWithValue = FlowNode<NodeData, "dataNode">;

export const isFlowNodeWithValue = (
  node: FlowNode
): node is FlowNodeWithValue => {
  return (node as FlowNodeWithValue).data.value != undefined;
};
