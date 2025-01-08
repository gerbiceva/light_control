import { client } from "../grpc/grpcClient";
import { Node as FlowNode, Edge as FlowEdge } from "@xyflow/react";
import { notifError } from "../utils/notifications";
import { isFlowNodeWithValue } from "../flow/Nodes/BaseNodes/utils/inputNodeType";

export const sync = (nodes: FlowNode[], edges: FlowEdge[]) => {
  return new Promise<void>((resolve, reject) => {
    client
      .graphUpdate({
        nodes: nodes.map((n) => {
          if (isFlowNodeWithValue(n)) {
            return {
              id: n.id,
              name: n.type || "",
              value: (n.data.value as object).toString(),
            };
          }
          return {
            id: n.id,
            name: n.type || "",
          };
        }),
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
