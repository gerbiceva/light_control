import { GrpcWebFetchTransport } from "@protobuf-ts/grpcweb-transport";
import { MyServiceClient } from "./client_code/service.client";

const transport = new GrpcWebFetchTransport({
  baseUrl: "http://localhost:50051",
  format: "binary",
  meta: {
    token: "ojla",
  },
});

export const client = new MyServiceClient(transport);
