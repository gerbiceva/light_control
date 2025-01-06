import { Connection, Edge } from "@xyflow/react";
import { getConnectionProperties } from "./typesFromConnection";

export const isValidConnection = (edge: Edge | Connection) => {
  const t = getConnectionProperties(edge);
  return t.isSameType && t.targetLen == 0;
};
