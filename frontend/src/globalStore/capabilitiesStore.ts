// store/users.ts
import { atom, computed } from "nanostores";
import { testCapabilitiesList } from "../flow/Nodes/ComputeNodes/test";
import { baseCapabilities } from "../flow/Nodes/BaseNodes/utils/baseCapabilities";
import { NodeCapability } from "../grpc/client_code/service";

export const $serverCapabilities = atom<NodeCapability[]>(testCapabilitiesList);

export const $capabilities = computed($serverCapabilities, (cap) => {
  const out: NodeCapability[] = [...cap, ...baseCapabilities];
  return out;
});

export function setCapabilites(capabilities: NodeCapability[]) {
  $serverCapabilities.set(capabilities);
}

setCapabilites(testCapabilitiesList);

// // initially get capabilities
// client.getCapabilities({}).then(
//   (data) => {
//     setCapabilites(data.response.nodes);
//     notifSuccess({
//       title: "Capabilities initialized",
//       message: "Node information loaded from server. App is ready to use",
//     });
//   },
//   (status) => {
//     notifError({
//       title: "Could not load capabilities! Check backend connection",
//       message: JSON.stringify(status, null, 2),
//     });
//     console.error(status);
//   }
// );
