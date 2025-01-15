import { FinalConnectionState } from "@xyflow/react";
import {
  $flowInst,
  addNode,
  generateFlowId,
  setEdges,
} from "../../../../globalStore/flowStore";
import { $mousePos } from "../../../../globalStore/mouseStore";
import { getNodeNamespaceAndTypeFromBaseType } from "./RegisterNodes";
import { getPortFromNode } from "../../../Edges/typesFromConnection";
import { addColoredEdge } from "../../../Edges/addColoredEdge";

export const addInputOnEdgeDrop = (
  _event: MouseEvent | TouchEvent,
  connectionState: FinalConnectionState
) => {
  // when a connection is dropped on the pane it's not valid
  if (
    !connectionState.isValid &&
    connectionState.fromHandle?.type == "target"
  ) {
    // console.log(getTargetPortFromNode(connectionState.));
    if (
      connectionState.fromHandle == null ||
      connectionState.fromNode == null ||
      connectionState.fromHandle == null ||
      !connectionState.fromHandle.id
    ) {
      return;
    }

    const port = getPortFromNode(
      connectionState.fromHandle,
      connectionState.fromNode,
      "source"
    );

    if (!port) {
      return;
    }

    const nodeType = getNodeNamespaceAndTypeFromBaseType(port?.type);
    if (!nodeType) {
      return;
    }

    const pos = $flowInst.get()?.screenToFlowPosition($mousePos.get()) || {
      x: 0,
      y: 0,
    };

    const id = generateFlowId();
    const origin: [number, number] = [1, 0.5];
    const newNode = {
      id,
      type: nodeType.namespaced,
      position: pos,
      data: { value: "" },
      origin,
    };

    addNode(newNode);

    setEdges(
      addColoredEdge({
        source: id,
        sourceHandle: nodeType.type,
        target: connectionState.fromNode?.id,
        targetHandle: connectionState.fromHandle?.id,
      })
    );
  }
};
