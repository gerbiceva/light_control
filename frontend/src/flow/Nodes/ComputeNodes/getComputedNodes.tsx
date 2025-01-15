import { $serverCapabilities } from "../../../globalStore/capabilitiesStore";
import { mergeNamespaceAndType } from "../../../sync/namespaceUtils";
import { generateComputeNodeFromCapability } from "./ComputeNodeFactory";
import { NodeTypes } from "@xyflow/react";

export const getComputedNodes = () => {
  const capabilities = $serverCapabilities;

  const customNodes: NodeTypes = {};

  capabilities.get().forEach((cap) => {
    const namespaceAndName = mergeNamespaceAndType(cap.namespace, cap.name);
    customNodes[namespaceAndName] = generateComputeNodeFromCapability(cap);
  });

  return customNodes;
};
