import { CustomFlowEdge, CustomFlowNode } from "../flow/Nodes/CustomNodeType";
import { getCapabilityFromNameNamespace } from "../globalStore/capabilitiesStore";
import { generateFlowId } from "../globalStore/flowStore";
import {
  mergeNamespaceAndType,
  splitTypeAndNamespace,
} from "../sync/namespaceUtils";
import { YEdge, YNode } from "./mixedStoreDto";

// NODES
export const yToFlowNode = (ynode: YNode): CustomFlowNode | undefined => {
  const cap = getCapabilityFromNameNamespace(ynode.name, ynode.namespace);
  if (!cap) {
    return;
  }

  return {
    data: {
      capability: cap,
      value: ynode.value,
    },
    id: ynode.id,
    position: ynode.position,
    type: mergeNamespaceAndType(ynode.namespace, ynode.name),
  };
};

export const flowToYNode = (node: CustomFlowNode): YNode | undefined => {
  const { namespace, type } = splitTypeAndNamespace(node.type!);
  return {
    id: node.id,
    name: type,
    namespace,
    position: node.position,
    state: "idle",
    value: node.data.value,
  };
};

// EDGES
export const yToFlowEdge = (edge: YEdge): CustomFlowEdge => {
  return {
    id: generateFlowId(),
    source: edge.fromNodeId,
    sourceHandle: edge.fromNodeHandle,
    target: edge.toNodeId,
    targetHandle: edge.toNodeHandle,
  };
};

export const flowToYEdge = (edge: CustomFlowEdge): YEdge => {
  return {
    fromNodeId: edge.source,
    fromNodeHandle: edge.sourceHandle!,
    toNodeId: edge.target,
    toNodeHandle: edge.targetHandle!,
  };
};
