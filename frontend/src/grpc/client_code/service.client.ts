// @generated by protobuf-ts 2.9.4 with parameter optimize_code_size
// @generated from protobuf file "service.proto" (syntax proto3)
// tslint:disable
import type { RpcTransport } from "@protobuf-ts/runtime-rpc";
import type { ServiceInfo } from "@protobuf-ts/runtime-rpc";
import { MyService } from "./service";
import type { GraphUpdated } from "./service";
import { stackIntercept } from "@protobuf-ts/runtime-rpc";
import type { Capabilities } from "./service";
import type { Void } from "./service";
import type { UnaryCall } from "@protobuf-ts/runtime-rpc";
import type { RpcOptions } from "@protobuf-ts/runtime-rpc";
/**
 * @generated from protobuf service MyService
 */
export interface IMyServiceClient {
    /**
     * get the list of nodes that the server supports along with their descriptions
     *
     * @generated from protobuf rpc: GetCapabilities(Void) returns (Capabilities);
     */
    getCapabilities(input: Void, options?: RpcOptions): UnaryCall<Void, Capabilities>;
    /**
     * Get the new edges and nodes from the frontend
     *
     * @generated from protobuf rpc: NodesUpdated(GraphUpdated) returns (Void);
     */
    nodesUpdated(input: GraphUpdated, options?: RpcOptions): UnaryCall<GraphUpdated, Void>;
}
/**
 * @generated from protobuf service MyService
 */
export class MyServiceClient implements IMyServiceClient, ServiceInfo {
    typeName = MyService.typeName;
    methods = MyService.methods;
    options = MyService.options;
    constructor(private readonly _transport: RpcTransport) {
    }
    /**
     * get the list of nodes that the server supports along with their descriptions
     *
     * @generated from protobuf rpc: GetCapabilities(Void) returns (Capabilities);
     */
    getCapabilities(input: Void, options?: RpcOptions): UnaryCall<Void, Capabilities> {
        const method = this.methods[0], opt = this._transport.mergeOptions(options);
        return stackIntercept<Void, Capabilities>("unary", this._transport, method, opt, input);
    }
    /**
     * Get the new edges and nodes from the frontend
     *
     * @generated from protobuf rpc: NodesUpdated(GraphUpdated) returns (Void);
     */
    nodesUpdated(input: GraphUpdated, options?: RpcOptions): UnaryCall<GraphUpdated, Void> {
        const method = this.methods[1], opt = this._transport.mergeOptions(options);
        return stackIntercept<GraphUpdated, Void>("unary", this._transport, method, opt, input);
    }
}
