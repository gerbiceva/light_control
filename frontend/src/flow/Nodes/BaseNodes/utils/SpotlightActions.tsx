import { Avatar } from "@mantine/core";
import { getColorFromEnum } from "../../../../utils/colorUtils";
import { Node } from "@xyflow/react";
import {
  $flowInst,
  addNode,
  generateFlowId,
} from "../../../../globalStore/flowStore";
import { baseCapabilities } from "./baseCapabilities";
import { CustomSpotData } from "../../../../views/Spotlight/CustomSpot/CustomSpotData";
import { $mousePos } from "../../../../globalStore/mouseStore";

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

export const generateNodeInstFromInput = (type: string): Node => {
  const pos = $flowInst.get()?.screenToFlowPosition($mousePos.get());

  return {
    id: generateFlowId(),
    type: type,
    position: pos || {
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
