import { $capabilities } from "../../../globalStore/capabilitiesStore";
import { generateComputeNodeFromCapability } from "./ComputeNodeFactory";
import { NodeTypes } from "@xyflow/react";

export const getComputedNodes = () => {
  const capabilities = $capabilities;

  const customNodes: NodeTypes = {};

  capabilities.get().forEach((cap) => {
    customNodes[cap.id] = generateComputeNodeFromCapability(cap);
  });

  return customNodes;
};
