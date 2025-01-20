import { atom } from "nanostores";
import { BaseType, NodeCapability } from "../grpc/client_code/service";
import { Handle } from "@xyflow/react";

// filter properties
export interface SpotFilter {
  type: "source" | "target";
  dataType: BaseType;
  fromCap: NodeCapability;
  fromHandle: Handle;
}
export const $spotFilter = atom<SpotFilter | undefined>(undefined);
export const setSpotFilter = (filter: SpotFilter) => {
  $spotFilter.set(filter);
};

export const resetSpotFilter = () => {
  $spotFilter.set(undefined);
};
