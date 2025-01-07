import { client } from "../grpc/grpcClient";
import { Node as FlowNode, Edge as FlowEdge } from "@xyflow/react";
import { notifError } from "../utils/notifications";

export const sync = (nodes: FlowNode[], edges: FlowEdge[]) => {
  return new Promise<void>((resolve, reject) => {
    client
      .graphUpdate({
        nodes: nodes.map((n) => ({
          id: n.id,
          name: n.type || "",
          value: "",
        })),
        edges: edges.map((e) => ({
          fromNode: e.source,
          fromPort: e.sourceHandle || "",
          toNode: e.target,
          toPort: e.targetHandle || "",
        })),
      })
      .then(
        (res) => {
          if (res.response) {
            resolve();
          }
        },
        (status) => {
          reject(status);
          notifError({
            title: "Cant send graph update",
            message: JSON.stringify(status, null, 1),
          });
          console.error(status);
        }
      );
  });
};
