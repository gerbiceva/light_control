import { Edge, Connection, Handle, Node as FlowNode } from "@xyflow/react";
import { $edges, $nodes } from "../../globalStore/flowStore";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { Port } from "../../grpc/client_code/service";
import { splitTypeAndNamespace } from "../../sync/namespaceUtils";

export const getCapFromNode = (node: FlowNode | null) => {
  const capabilities = $capabilities.get();

  const { namespace: nsFrom, type: tFrom } = splitTypeAndNamespace(
    node?.type || ""
  );
  const fromCap = capabilities.find(
    (cap) => cap.name == tFrom && nsFrom == cap.namespace
  );

  return fromCap;
};

export const getPortFromNode = (
  handle: Handle | null,
  node: FlowNode | null,
  type: "source" | "target"
) => {
  if (!handle || !node) {
    return;
  }
  const fromCap = getCapFromNode(node);

  if (type == "target") {
    return fromCap?.inputs.find((port) => port.name == handle.id);
  }
  if (type == "source") {
    return fromCap?.outputs.find((port) => port.name == handle.id);
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
): ConnectionTypesOut => {
  const nodes = $nodes.get();
  const edges = $edges.get();
  //TODO: make an override that does not loop over nodes and gets a node from input. Use with AddInputOnEdgeDrop

  // get from node, namespace, type, capability and port
  const from = nodes.find((node) => node.id == edge.source);

  const fromCap = from ? getCapFromNode(from) : undefined;
  const fromPort = fromCap?.outputs.find(
    (cap) => cap.name == edge.sourceHandle
  );

  // get to node, namespace, type, capability and port
  const to = nodes.find((node) => node.id == edge.target);
  const toCap = to ? getCapFromNode(to) : undefined;
  const toPort = toCap?.inputs.find((cap) => cap.name == edge.targetHandle);
  const targetEdges = edges.filter(
    (edge) => edge.target == to?.id && edge.targetHandle == toPort?.name
  );

  return {
    from: fromPort,
    to: toPort,
    isSameType: fromPort?.type == toPort?.type || false,
    targetLen: targetEdges.length,
  };
};
