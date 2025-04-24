import * as Y from "yjs";

/**
 * Converts a value to a shallow Y.Map.
 * - For objects: creates a Y.Map with all enumerable own properties
 * - For arrays: creates a Y.Map with index-number keys
 * - For primitives: creates an empty Y.Map (or could wrap in a special key)
 *
 * @param value - The value to convert to a Y.Map
 * @returns A new Y.Map containing the shallow representation
 */
export function toShallowYMap<T>(value: T): Y.Map<T> {
  const yMap = new Y.Map<T>();

  if (value === null || value === undefined) {
    throw new Error("Empty or undefined " + value);
  }

  if (typeof value === "object") {
    // Handle arrays (convert to {0: val, 1: val, ...})
    if (Array.isArray(value)) {
      value.forEach((item, index) => {
        yMap.set(index.toString(), item);
      });
    }
    // Handle plain objects
    else {
      Object.entries(value).forEach(([key, val]) => {
        yMap.set(key, val);
      });
    }
  }
  // Optionally handle primitives by wrapping them
  // else {
  //     yMap.set('value', value);
  // }

  return yMap;
}
