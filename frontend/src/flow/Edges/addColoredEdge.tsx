import { addEdge, Connection, Edge } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";

export const addColoredEdge = (edgeParams: Connection, edges: Edge[]) => {
  const t = getConnectionProperties(edgeParams);

  return addEdge(edgeParams, edges).map((edge) => {
    return {
      style: {
        stroke: getColorFromEnum(t.from?.type || 0)[4],
        strokeWidth: 3,
      },
      ...edge,
    };
  });
};
