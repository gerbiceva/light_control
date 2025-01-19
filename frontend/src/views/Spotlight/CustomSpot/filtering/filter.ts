import { SpotFilter } from "../../../../globalStore/spotlightFilterStore";
import { CustomSpotData } from "../CustomSpotData";

const filter = (candidate: string, query: string): boolean => {
  return candidate
    .toLocaleLowerCase()
    .includes(query.trim().toLocaleLowerCase());
};

export const filterItemsByName = <T>(
  items: T[],
  serialize: (item: T) => string,
  query: string
): T[] => {
  const out: T[] = [];

  for (const item of items) {
    const serialized = serialize(item);
    if (filter(serialized, query)) {
      out.push(item);
    }
  }

  return out;
};

export const filterItemsByType = (
  items: CustomSpotData[],
  filter: SpotFilter
): CustomSpotData[] => {
  const out: CustomSpotData[] = [];
  for (const item of items) {
    if (filter.type == "target") {
      if (
        item.capability.outputs.filter((port) => port.type == filter.dataType)
          .length > 0
      ) {
        out.push(item);
      }
    }
    if (filter.type == "source") {
      if (
        item.capability.inputs.filter((port) => port.type == filter.dataType)
          .length > 0
      ) {
        out.push(item);
      }
    }
  }

  return out;
};
