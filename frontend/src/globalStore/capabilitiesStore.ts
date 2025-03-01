// store/users.ts
import { atom, computed } from "nanostores";
import { NodeCapability } from "../grpc/client_code/service";
import { client } from "../grpc/grpcClient";
import { notifSuccess, notifError } from "../utils/notifications";
import { primitiveCapabilities } from "../flow/Nodes/BaseNodes/utils/baseCapabilities";

export const $serverCapabilities = atom<NodeCapability[]>([]);

export const $capabilities = computed($serverCapabilities, (cap) => {
  const out: NodeCapability[] = [...cap, ...primitiveCapabilities];
  return out;
});

export function setCapabilites(capabilities: NodeCapability[]) {
  $serverCapabilities.set(capabilities);
}

// initially get capabilities
client.getCapabilities({}).then(
  (data) => {
    setCapabilites(data.response.nodes);
    notifSuccess({
      title: "Capabilities initialized",
      message: "Node information loaded from server. App is ready to use",
    });
  },
  (status) => {
    notifError({
      title: "Could not load capabilities! Check backend connection",
      message: JSON.stringify(status, null, 2),
    });
    console.error(status);
  }
);

export const getCapabilityFromNameNamespace = (
  name: string,
  namespace: string
): NodeCapability | undefined => {
  return $serverCapabilities
    .get()
    .find((cap) => cap.name == name && cap.namespace == namespace);
};
