import { Connection, Edge } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";
import { generateFlowId } from "../../globalStore/flowStore";

export const getColoredEdge = (edgeParams: Connection) => {
  const t = getConnectionProperties(edgeParams);
  if (!t) {
    return;
  }

  const edge: Edge = {
    id: generateFlowId(),
    source: edgeParams.source,
    sourceHandle: edgeParams.sourceHandle,
    target: edgeParams.target,
    targetHandle: edgeParams.targetHandle,
    style: {
      stroke: getColorFromEnum(t.from?.type || t.to?.type || 0)[4],
      strokeWidth: 3,
    },
  };

  return edge;
};
