import { client } from "../grpc/grpcClient";
import { Edge as FlowEdge } from "@xyflow/react";
import { notifError } from "../utils/notifications";
import { isFlowNodeWithValue } from "../flow/Nodes/BaseNodes/utils/inputNodeType";
import { splitTypeAndNamespace } from "./namespaceUtils";
import { CustomFlowNode } from "../flow/Nodes/CustomNodeType";

export const sync = (nodes: CustomFlowNode[], edges: FlowEdge[]) => {
  return new Promise<void>((resolve, reject) => {
    resolve();
    client
      .graphUpdate({
        nodes: nodes.map((n) => {
          const { namespace, type } = splitTypeAndNamespace(n.type || "");
          if (isFlowNodeWithValue(n)) {
            return {
              id: n.id,
              name: type,
              value: (n.data.value as object).toString(),
              namespace,
            };
          }
          return {
            id: n.id,
            name: type,
            namespace,
          };
        }),
        edges: edges.map((e) => {
          return {
            fromNamespace: "",
            toNamespace: "",
            fromNode: e.source,
            fromPort: e.sourceHandle || "",
            toNode: e.target,
            toPort: e.targetHandle || "",
          };
        }),
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
        },
      );
  });
};
