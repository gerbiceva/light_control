import { SpotlightActionData } from "@mantine/spotlight";
import { NodeCapability } from "../../../grpc/client_code/service";

export interface CustomSpotData extends SpotlightActionData {
  capability: NodeCapability;
}
