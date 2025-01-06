// store/users.ts
import { atom, computed } from "nanostores";
import { Node } from "../grpc/client_code/service";
import { testCapabilitiesList } from "../flow/Nodes/ComputeNodes/test";
import { baseCapabilities } from "../flow/Nodes/BaseNodes/utils/baseCapabilities";

export const $serverCapabilities = atom<Node[]>(testCapabilitiesList);

export const $capabilities = computed($serverCapabilities, (cap) => {
  return [...cap, ...baseCapabilities];
});

export function setCapabilites(capabilities: Node[]) {
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
