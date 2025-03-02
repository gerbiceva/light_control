import { addEdge, Connection } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";
import { getColorFromEnum } from "../../utils/colorUtils";
import { getActiveYgraph } from "../../crdt/repo";

export const addColoredEdge = (edgeParams: Connection) => {
  const t = getConnectionProperties(edgeParams);
  const g = getActiveYgraph();
  const edges = g.edges;

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
