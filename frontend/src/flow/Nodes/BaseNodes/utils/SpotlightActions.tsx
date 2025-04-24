import { Avatar } from "@mantine/core";
import { getColorFromEnum } from "../../../../utils/colorUtils";
import { $flowInst, generateFlowId } from "../../../../globalStore/flowStore";
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
import { addEdge, addNode } from "../../../../crdt/repo";
import { $spotFilter } from "../../../../globalStore/spotlightFilterStore";
import { getColoredEdge } from "../../../Edges/getColoredEdge";
import { NodeCapability } from "../../../../grpc/client_code/service";

export const inputNodesActions: CustomSpotData[] = primitiveCapabilities.map(
  (cap, i) => ({
    id: cap.name,
    label: cap.name,
    description: cap.description,
    onClick: () => {
      createNewNode(cap, i);
    },
    capability: cap,
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(cap.outputs[0].type)[5]}>
        {cap.name.slice(0, 3)}
      </Avatar>
    ),
  })
);

const createNewNode = (cap: NodeCapability, capIndex: number) => {
  const spotFilter = $spotFilter.get();
  const node = generateNodeInstFromInput(
    mergeNamespaceAndType("primitive", cap.name)
  );

  if (spotFilter && cap) {
    addEdge(
      spotFilter.type == "target"
        ? getColoredEdge({
            source: node.id,
            sourceHandle: capIndex.toString(),
            target: spotFilter.fromHandle.nodeId,
            targetHandle: spotFilter.fromHandle.id!,
          })
        : getColoredEdge({
            source: spotFilter.fromHandle.nodeId,
            sourceHandle: spotFilter.fromHandle.id!,
            target: node.id,
            targetHandle: capIndex.toString(),
          })
    );
  }
  addNode(node);
};
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
