// store/users.ts
import { atom } from "nanostores";
import { Node } from "../grpc/client_code/service";
import { testCapabilitiesList } from "../flow/Nodes/ComputeNodes/test";

export const $capabilities = atom<Node[]>(testCapabilitiesList);

export function setCapabilites(capabilities: Node[]) {
  $capabilities.set(capabilities);
}
