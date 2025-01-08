import { addEdge, Connection } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";
import { $edges } from "../../globalStore/flowStore";

export const addColoredEdge = (edgeParams: Connection) => {
  const t = getConnectionProperties(edgeParams);

  return addEdge(edgeParams, $edges.get()).map((edge) => {
    return {
      style: {
        stroke: getColorFromEnum(t.from?.type || 0)[4],
        strokeWidth: 3,
      },
      ...edge,
    };
  });
};
