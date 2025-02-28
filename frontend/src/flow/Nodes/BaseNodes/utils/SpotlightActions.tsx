import { Avatar } from "@mantine/core";
import { getColorFromEnum } from "../../../../utils/colorUtils";
import {
  $flowInst,
  addNode,
  generateFlowId,
} from "../../../../globalStore/flowStore";
import {
  getBaseCapabilityFromType,
  primitiveCapabilities,
} from "./baseCapabilities";
import { CustomSpotData } from "../../../../views/Spotlight/CustomSpot/CustomSpotData";
import { $frozenMousePos } from "../../../../globalStore/mouseStore";
import {
  mergeNamespaceAndType,
  splitTypeAndNamespace,
} from "../../../../sync/namespaceUtils";
import { CustomFlowNode } from "../../CustomNodeType";

export const inputNodesActions: CustomSpotData[] = primitiveCapabilities.map(
  (cap) => ({
    id: cap.name,
    label: cap.name,
    description: cap.description,
    onClick: () =>
      addNode(
        generateNodeInstFromInput(mergeNamespaceAndType("primitive", cap.name)),
      ),
    capability: cap,
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(cap.outputs[0].type)[5]}>
        {cap.name.slice(0, 3)}
      </Avatar>
    ),
  }),
);

export const generateNodeInstFromInput = (type: string): CustomFlowNode => {
  const pos = $flowInst.get()?.screenToFlowPosition($frozenMousePos.get());
  const { type: strippedType } = splitTypeAndNamespace(type);
  const capability = getBaseCapabilityFromType(strippedType);
  if (!capability) {
    throw new Error(`Capability not found for type ${type}`);
  }

  return {
    id: generateFlowId(),
    type: type,
    position: pos || {
      x: 0,
      y: 0,
    },
    data: {
      capability: capability,
    },
  };
};
