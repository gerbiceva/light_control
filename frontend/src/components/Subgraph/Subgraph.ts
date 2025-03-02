import {
  CustomFlowEdge,
  CustomFlowNode,
} from "../../flow/Nodes/CustomNodeType";

export interface SubGraph {
  id: number;
  name: string;
  description?: string;
  nodes: CustomFlowNode[];
  edges: CustomFlowEdge[];
}

export const mainFlow: SubGraph = {
  id: 0,
  name: "main",
  description: "This main flow gets executed always",
  nodes: [],
  edges: [],
};
