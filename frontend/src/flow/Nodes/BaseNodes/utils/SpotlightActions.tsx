import { Avatar } from "@mantine/core";
import { getColorFromEnum } from "../../../../utils/colorUtils";
import { Node } from "@xyflow/react";
import { addNode, generateFlowId } from "../../../../globalStore/flowStore";
import { baseCapabilities } from "./baseCapabilities";
import { CustomSpotData } from "../../../../views/Spotlight/CustomSpot/CustomSpotData";

export const inputNodesActions: CustomSpotData[] = baseCapabilities.map(
  (cap) => ({
    id: cap.name,
    label: cap.name,
    description: cap.description,
    onClick: () => addNode(generateNodeInstFromInput(cap.name)),
    capability: cap,
    leftSection: (
      <Avatar radius={0} color={getColorFromEnum(cap.outputs[0].type)[5]}>
        {cap.name.slice(0, 3)}
      </Avatar>
    ),
  })
);

export const generateNodeInstFromInput = (
  type: string
  // inst: ReactFlowInstance<FlowNode, Edge>
): Node => {
  return {
    id: generateFlowId(),
    type: type,
    position: {
      x: 0,
      y: 0,
    },
    // position: inst.screenToFlowPosition({
    //   x: window.clientX,
    //   y: clientY,
    // }),
    data: {},
  };
};
