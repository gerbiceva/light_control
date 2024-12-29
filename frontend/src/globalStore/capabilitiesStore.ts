// store/users.ts
import { atom } from "nanostores";
import { Node } from "../grpc/client_code/service";
import { testCapabilitiesList } from "../flow/Nodes/ComputeNodes/test";

export const $capabilities = atom<Node[]>(testCapabilitiesList);

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

export function setCapabilites(capabilities: Node[]) {
  $capabilities.set(capabilities);
}

setCapabilites(testCapabilitiesList);
