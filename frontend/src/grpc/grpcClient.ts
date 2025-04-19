import { GrpcWebFetchTransport } from "@protobuf-ts/grpcweb-transport";
import { MyServiceClient } from "./client_code/service.client";

const transport = new GrpcWebFetchTransport({
  baseUrl: `http://${window.location.hostname}:50051`,
  format: "binary",
  // meta: {
  //   token: "ojla",
  // },
});

export const client = new MyServiceClient(transport);

// const notifs = client.streamNotifications({},{});
// notifs.then(
//   (notif) => {
//     console.log(notif.);
//   },
//   (err) => {
//     console.error({ err });
//   }
// );
