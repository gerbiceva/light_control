import { atom } from "nanostores";
import { BaseType } from "../grpc/client_code/service";

export interface SpotFilter {
  type: "source" | "target";
  dataType: BaseType;
}

export const $spotFilter = atom<SpotFilter | undefined>(undefined);

export const setSpotFilter = (filter: SpotFilter) => {
  $spotFilter.set(filter);
};

export const resetSpotFilter = () => {
  $spotFilter.set(undefined);
};
