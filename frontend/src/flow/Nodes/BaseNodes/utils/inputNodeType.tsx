import { Node as FlowNode, NodeProps } from "@xyflow/react";

type NodeData = {
  data: {
    value: unknown;
  };
};
export type FlowNodeWithValue = FlowNode<NodeData, "dataNode">;

export const isFlowNodeWithValue = (
  node: NodeProps<FlowNode>,
): node is NodeProps<FlowNodeWithValue> => {
  return (node as NodeProps<FlowNodeWithValue>).selected?.data != undefined;
};
