import { NodeTypes } from "@xyflow/react";
import { ColorNode } from "../ColorNode";
import { CurveNode } from "../CurveNode";
import { FloatNode } from "../FloatNode";
import { IntNode } from "../IntNode";
import { StringNode } from "../StringNode";

export const inputNodes: NodeTypes = {
  Color: ColorNode,
  Int: IntNode,
  Float: FloatNode,
  String: StringNode,
  Curve: CurveNode,
};
