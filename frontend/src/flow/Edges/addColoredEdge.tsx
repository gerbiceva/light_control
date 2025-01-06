import { addEdge, Connection, Edge } from "@xyflow/react";
import { getTypesFromConnection } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";

export const addColoredEdge = (edgeParams: Connection, edges: Edge[]) => {
  const t = getTypesFromConnection(edgeParams);

  return addEdge(edgeParams, edges).map((edge) => {
    return {
      style: {
        stroke: getColorFromEnum(t[0]?.type || 0)[4],
        strokeWidth: 3,
      },
      ...edge,
    };
  });
};
