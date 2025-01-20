import { spotlight } from "@mantine/spotlight";
import { FinalConnectionState } from "@xyflow/react";
import { freezeMousePos } from "../../../../globalStore/mouseStore";
import { setSpotFilter } from "../../../../globalStore/spotlightFilterStore";
import {
  getCapFromNode,
  getPortFromNode,
} from "../../../Edges/typesFromConnection";

export const addInputOnEdgeDrop = (
  _event: MouseEvent | TouchEvent,
  connectionState: FinalConnectionState
) => {
  // when a connection is dropped on the pane it's not valid
  if (!connectionState.isValid) {
    if (
      !connectionState.fromHandle ||
      connectionState.fromNode == null ||
      !connectionState.fromHandle.id
    ) {
      return;
    }
    const port = getPortFromNode(
      connectionState.fromHandle,
      connectionState.fromNode,
      connectionState.fromHandle?.type
    );

    const cap = getCapFromNode(connectionState.fromNode);

    if (!port || !cap) {
      return;
    }

    // const nodeType = getNodeNamespaceAndTypeFromBaseType(port?.type);
    // if (
    //   nodeType != undefined &&
    //   connectionState.fromHandle?.type == "target" &&
    //   false
    // ) {
    //   // generate a primitive node
    //   const pos = $flowInst
    //     .get()
    //     ?.screenToFlowPosition($frozenMousePos.get()) || {
    //     x: 0,
    //     y: 0,
    //   };

    //   const id = generateFlowId();
    //   const origin: [number, number] = [1, 0.5];
    //   const newNode = {
    //     id,
    //     type: nodeType.namespaced,
    //     position: pos,
    //     data: { value: "" },
    //     origin,
    //   };

    //   addNode(newNode);

    //   setEdges(
    //     addColoredEdge({
    //       source: id,
    //       sourceHandle: nodeType.type,
    //       target: connectionState.fromNode?.id,
    //       targetHandle: connectionState.fromHandle?.id,
    //     })
    //   );
    // } else {
    freezeMousePos();
    setSpotFilter({
      type: connectionState.fromHandle?.type,
      dataType: port.type,
      fromCap: cap,
      fromHandle: connectionState.fromHandle,
    });
    spotlight.open();
  }
};
