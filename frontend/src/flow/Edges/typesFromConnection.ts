import { Edge, Connection, Handle, Node as FlowNode } from "@xyflow/react";
import { $edges, $nodes } from "../../globalStore/flowStore";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { Port } from "../../grpc/client_code/service";
import { splitTypeAndNamespace } from "../../sync/namespaceUtils";

export const getPortFromNode = (
  handle: Handle | null,
  node: FlowNode | null,
  type: "source" | "target"
) => {
  if (!handle || !node) {
    return;
  }
  const capabilities = $capabilities.get();

  const { namespace: nsFrom, type: tFrom } = splitTypeAndNamespace(
    node?.type || ""
  );

  const fromCap = capabilities.find(
    (cap) => cap.name == tFrom && nsFrom == cap.namespace
  );

  if (type == "source") {
    return fromCap?.inputs.find((cap) => cap.name == handle.id);
  }
  if (type == "target") {
    return fromCap?.outputs.find((cap) => cap.name == handle.id);
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
  const capabilities = $capabilities.get();

  // get from node, namespace, type, capability and port
  const from = nodes.find((node) => node.id == edge.source);
  const { namespace: nsFrom, type: tFrom } = splitTypeAndNamespace(
    from?.type || ""
  );
  const fromCap = capabilities.find(
    (cap) => cap.name == tFrom && nsFrom == cap.namespace
  );
  const fromPort = fromCap?.outputs.find(
    (cap) => cap.name == edge.sourceHandle
  );

  // get to node, namespace, type, capability and port
  const to = nodes.find((node) => node.id == edge.target);
  const { namespace: nsTo, type: tTo } = splitTypeAndNamespace(to?.type || "");
  const toCap = capabilities.find(
    (cap) => cap.name == tTo && nsTo == cap.namespace
  );
  const toPort = toCap?.inputs.find((cap) => cap.name == edge.targetHandle);
  const targetEdges = edges.filter(
    (edge) => edge.target == to?.id && edge.targetHandle == toPort?.name
  );

  // console.log({ from }, { fromCap }, { toCap });
  // console.log({ fromPort }, { toPort });

  return {
    from: fromPort,
    to: toPort,
    isSameType: fromPort?.type == toPort?.type || false,
    targetLen: targetEdges.length,
  };
};
