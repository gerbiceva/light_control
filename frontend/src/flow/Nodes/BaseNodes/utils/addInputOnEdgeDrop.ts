import { spotlight } from "@mantine/spotlight";
import { FinalConnectionState } from "@xyflow/react";
import { freezeMousePos } from "../../../../globalStore/mouseStore";
import { setSpotFilter } from "../../../../globalStore/spotlightFilterStore";
import { getPortFromNode } from "../../../Edges/typesFromConnection";
import { isCustomFlowNode } from "../../CustomNodeType";

export const addInputOnEdgeDrop = (
  _event: MouseEvent | TouchEvent,
  connectionState: FinalConnectionState
) => {
  if (
    !connectionState.fromNode ||
    !isCustomFlowNode(connectionState.fromNode)
  ) {
    return;
  }

  // when a connection is dropped on the pane it's not valid
  if (connectionState.isValid) {
    return;
  }
  // dropped on pane

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

  const cap = connectionState.fromNode.data.capability;

  if (!port || !cap) {
    return;
  }

  freezeMousePos();
  console.log("spotlight filter:", {
    type: connectionState.fromHandle?.type,
    dataType: port.type,
    fromCap: cap,
    fromHandle: connectionState.fromHandle,
  });

  setSpotFilter({
    type: connectionState.fromHandle?.type,
    dataType: port.type,
    fromCap: cap,
    fromHandle: connectionState.fromHandle,
  });
  spotlight.open();
};
