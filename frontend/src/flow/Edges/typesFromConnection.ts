import { Edge, Connection, Handle } from "@xyflow/react";
import { Port } from "../../grpc/client_code/service";
import { CustomFlowNode } from "../Nodes/CustomNodeType";
import { getActiveYgraph } from "../../crdt/repo";

export const getPortFromNode = (
  handle: Handle | null,
  node: CustomFlowNode | null,
  type: "source" | "target"
) => {
  if (!handle || !node) {
    return;
  }

  const fromCap = node.data.capability;

  if (type == "target") {
    return fromCap?.inputs[parseInt(handle.id!)];
  }
  if (type == "source") {
    return fromCap?.outputs[parseInt(handle.id!)];
  }
};

interface ConnectionTypesOut {
  from?: Port;
  to?: Port;
  isSameType: boolean;
  targetLen: number;
}
export const getConnectionProperties = (
  edge: Edge | Connection
): ConnectionTypesOut | undefined => {
  const g = getActiveYgraph();
  if (!g) {
    return;
  }
  const nodes = g.nodes;
  const edges = g.edges;

  //TODO: make an override that does not loop over nodes and gets a node from input. Use with AddInputOnEdgeDrop
  // get from node, namespace, type, capability and port
  const from = nodes.find((node) => node.id == edge.source);

  const fromCap = from ? from.data.capability : undefined;
  const fromPort = fromCap?.outputs[parseInt(edge.sourceHandle!)];

  // get to node, namespace, type, capability and port
  const to = nodes.find((node) => node.id == edge.target);
  const toCap = to ? to.data.capability : undefined;
  const targetHandleIndex = edge.targetHandle!;
  const toPort = toCap?.inputs[parseInt(targetHandleIndex)];

  const targetEdges = edges.filter(
    (edge) => edge.target == to?.id && edge.targetHandle == targetHandleIndex
  );

  return {
    from: fromPort,
    to: toPort,
    isSameType: fromPort?.type == toPort?.type || false,
    targetLen: targetEdges.length,
  };
};
