import { Connection, Edge } from "@xyflow/react";
import { getTypesFromConnection } from "./typesFromConnection";

/**
 * @deprecated this needs further tuning
 * @param edge
 * @returns
 */
export const isValidConnection = (edge: Edge | Connection) => {
  const t = getTypesFromConnection(edge);
  if (!t[0] || !t[1]) {
    return false;
  }

  return t[0].type == t[1].type;
};
