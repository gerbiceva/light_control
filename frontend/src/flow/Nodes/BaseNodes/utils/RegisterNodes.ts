import { NodeTypes } from "@xyflow/react";
import { ColorNode } from "../ColorNode";
import { CurveNode } from "../CurveNode";
import { FloatNode } from "../FloatNode";
import { IntNode } from "../IntNode";
import { StringNode } from "../StringNode";
import { BaseType } from "../../../../grpc/client_code/service";

export const inputNodes: NodeTypes = {
  Color: ColorNode,
  Int: IntNode,
  Float: FloatNode,
  String: StringNode,
  Curve: CurveNode,
};

export type InputNodeTypes = keyof typeof inputNodes;

export const getNodeNamespaceAndTypeFromBaseType = (inp: BaseType) => {
  const nodeType = getNodeTypeFromBaseType(inp);
  if (nodeType == null) {
    return null;
  }

  return {
    namespaced: "primitive/" + nodeType,
    type: nodeType,
  };
};
export const getNodeTypeFromBaseType = (
  inp: BaseType
): InputNodeTypes | null => {
  switch (inp) {
    case BaseType.Color:
      return "Color";

    case BaseType.Int:
      return "Int";

    case BaseType.Float:
      return "Float";

    case BaseType.String:
      return "String";

    case BaseType.Curve:
      return "Curve";

    default:
      return null;
  }
};
