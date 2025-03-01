import { addEdge, Connection } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";
import { getYState } from "../../crdt/repo";

export const addColoredEdge = (edgeParams: Connection) => {
  const t = getConnectionProperties(edgeParams);
  const { main } = getYState();
  const edges = main.edges;

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
