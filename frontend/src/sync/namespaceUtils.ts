import { NodeTypes } from "@xyflow/react";

export const mergeNamespaceAndType = (namespace: string, type: string) => {
  return `${namespace}/${type}`;
};

export const splitTypeAndNamespace = (mergedType: string) => {
  const spl = mergedType.split("/");
  if (spl.length != 2) {
    throw new Error(`splitting failed. string: ${mergedType} is not valid`);
  }

  return {
    namespace: spl[0],
    type: spl[1],
  };
};

export const mapPrimitivesToNamespaced = (prim: NodeTypes): NodeTypes => {
  const out: NodeTypes = {};

  Object.keys(prim).forEach((primitiveName) => {
    const namespacedType = mergeNamespaceAndType("primitive", primitiveName);
    out[namespacedType] = prim[primitiveName];
  });

  return out;
};
