import { Edge, Connection } from "@xyflow/react";
import { $nodes } from "../../globalStore/flowStore";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { Port } from "../../grpc/client_code/service";

export const getTypesFromConnection = (
  edge: Edge | Connection
): [Port?, Port?] => {
  const nodes = $nodes.get();
  const capabilities = $capabilities.get();

  const from = nodes.find((node) => node.id == edge.source);
  const fromCap = capabilities.find((cap) => cap.id == from?.type);
  const fromPort = fromCap?.outputs.find(
    (cap) => cap.name == edge.sourceHandle
  );

  const to = nodes.find((node) => node.id == edge.target);
  const toCap = capabilities.find((cap) => cap.id == to?.type);
  const toPort = toCap?.inputs.find((cap) => cap.name == edge.targetHandle);

  return [fromPort, toPort];
};
