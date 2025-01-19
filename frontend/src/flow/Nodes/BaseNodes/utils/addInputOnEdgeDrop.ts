import { FinalConnectionState } from "@xyflow/react";
import {
  $flowInst,
  addNode,
  generateFlowId,
  setEdges,
} from "../../../../globalStore/flowStore";
import {
  $frozenMousePos,
  freezeMousePos,
} from "../../../../globalStore/mouseStore";
import { getNodeNamespaceAndTypeFromBaseType } from "./RegisterNodes";
import { getPortFromNode } from "../../../Edges/typesFromConnection";
import { addColoredEdge } from "../../../Edges/addColoredEdge";
import { setSpotFilter } from "../../../../globalStore/spotlightFilterStore";
import { spotlight } from "@mantine/spotlight";

export const addInputOnEdgeDrop = (
  _event: MouseEvent | TouchEvent,
  connectionState: FinalConnectionState
) => {
  // when a connection is dropped on the pane it's not valid
  if (!connectionState.isValid) {
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
      connectionState.fromHandle?.type
    );

    if (!port) {
      return;
    }

    console.log({ port });

    const nodeType = getNodeNamespaceAndTypeFromBaseType(port?.type);
    if (nodeType != undefined && connectionState.fromHandle?.type == "target") {
      // generate a primitive node
      const pos = $flowInst
        .get()
        ?.screenToFlowPosition($frozenMousePos.get()) || {
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
    } else {
      freezeMousePos();
      setSpotFilter({
        type: connectionState.fromHandle?.type,
        dataType: port.type,
      });
      spotlight.open();
    }
  }
};
