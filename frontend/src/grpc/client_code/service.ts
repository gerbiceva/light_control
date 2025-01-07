// @generated by protobuf-ts 2.9.4 with parameter optimize_code_size
// @generated from protobuf file "service.proto" (syntax proto3)
// tslint:disable
import { ServiceType } from "@protobuf-ts/runtime-rpc";
import { MessageType } from "@protobuf-ts/runtime";
/**
 * @generated from protobuf message Void
 */
export interface Void {
}
// Capabilities

/**
 * @generated from protobuf message Port
 */
export interface Port {
    /**
     * @generated from protobuf field: string name = 1;
     */
    name: string;
    /**
     * @generated from protobuf field: BaseType type = 3;
     */
    type: BaseType;
}
/**
 * @generated from protobuf message NodeCapability
 */
export interface NodeCapability {
    /**
     * @generated from protobuf field: string name = 1;
     */
    name: string;
    /**
     * @generated from protobuf field: string description = 2;
     */
    description: string;
    /**
     * @generated from protobuf field: repeated Port inputs = 4;
     */
    inputs: Port[];
    /**
     * @generated from protobuf field: repeated Port outputs = 5;
     */
    outputs: Port[];
}
/**
 * @generated from protobuf message Capabilities
 */
export interface Capabilities {
    /**
     * @generated from protobuf field: repeated NodeCapability nodes = 1;
     */
    nodes: NodeCapability[];
}
/**
 * graph state update
 * change from to something else
 *
 * @generated from protobuf message EdgeMsg
 */
export interface EdgeMsg {
    /**
     * @generated from protobuf field: string fromNode = 1;
     */
    fromNode: string;
    /**
     * @generated from protobuf field: string fromPort = 2;
     */
    fromPort: string;
    /**
     * @generated from protobuf field: string toNode = 3;
     */
    toNode: string;
    /**
     * @generated from protobuf field: string toPort = 4;
     */
    toPort: string;
}
/**
 * add data to node so we can share primitives
 *
 * @generated from protobuf message NodeMsg
 */
export interface NodeMsg {
    /**
     * @generated from protobuf field: string id = 1;
     */
    id: string;
    /**
     * @generated from protobuf field: string name = 2;
     */
    name: string;
    /**
     * @generated from protobuf field: string value = 3;
     */
    value: string;
}
/**
 * @generated from protobuf message GraphUpdated
 */
export interface GraphUpdated {
    /**
     * @generated from protobuf field: repeated NodeMsg nodes = 1;
     */
    nodes: NodeMsg[];
    /**
     * @generated from protobuf field: repeated EdgeMsg edges = 2;
     */
    edges: EdgeMsg[];
}
/**
 * @generated from protobuf enum BaseType
 */
export enum BaseType {
    /**
     * @generated from protobuf enum value: Int = 0;
     */
    Int = 0,
    /**
     * @generated from protobuf enum value: Float = 2;
     */
    Float = 2,
    /**
     * @generated from protobuf enum value: String = 3;
     */
    String = 3,
    /**
     * @generated from protobuf enum value: Color = 4;
     */
    Color = 4,
    /**
     * @generated from protobuf enum value: Curve = 5;
     */
    Curve = 5,
    /**
     * Non inputable
     *
     * @generated from protobuf enum value: ColorArray = 6;
     */
    ColorArray = 6,
    /**
     * @generated from protobuf enum value: Array = 7;
     */
    Array = 7,
    /**
     * @generated from protobuf enum value: Vector2D = 8;
     */
    Vector2D = 8,
    /**
     * @generated from protobuf enum value: Vector3D = 9;
     */
    Vector3D = 9
}
// @generated message type with reflection information, may provide speed optimized methods
class Void$Type extends MessageType<Void> {
    constructor() {
        super("Void", []);
    }
}
/**
 * @generated MessageType for protobuf message Void
 */
export const Void = new Void$Type();
// @generated message type with reflection information, may provide speed optimized methods
class Port$Type extends MessageType<Port> {
    constructor() {
        super("Port", [
            { no: 1, name: "name", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "type", kind: "enum", T: () => ["BaseType", BaseType] }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message Port
 */
export const Port = new Port$Type();
// @generated message type with reflection information, may provide speed optimized methods
class NodeCapability$Type extends MessageType<NodeCapability> {
    constructor() {
        super("NodeCapability", [
            { no: 1, name: "name", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "description", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 4, name: "inputs", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => Port },
            { no: 5, name: "outputs", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => Port }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message NodeCapability
 */
export const NodeCapability = new NodeCapability$Type();
// @generated message type with reflection information, may provide speed optimized methods
class Capabilities$Type extends MessageType<Capabilities> {
    constructor() {
        super("Capabilities", [
            { no: 1, name: "nodes", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => NodeCapability }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message Capabilities
 */
export const Capabilities = new Capabilities$Type();
// @generated message type with reflection information, may provide speed optimized methods
class EdgeMsg$Type extends MessageType<EdgeMsg> {
    constructor() {
        super("EdgeMsg", [
            { no: 1, name: "fromNode", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "fromPort", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "toNode", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 4, name: "toPort", kind: "scalar", T: 9 /*ScalarType.STRING*/ }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message EdgeMsg
 */
export const EdgeMsg = new EdgeMsg$Type();
// @generated message type with reflection information, may provide speed optimized methods
class NodeMsg$Type extends MessageType<NodeMsg> {
    constructor() {
        super("NodeMsg", [
            { no: 1, name: "id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "name", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "value", kind: "scalar", T: 9 /*ScalarType.STRING*/ }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message NodeMsg
 */
export const NodeMsg = new NodeMsg$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GraphUpdated$Type extends MessageType<GraphUpdated> {
    constructor() {
        super("GraphUpdated", [
            { no: 1, name: "nodes", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => NodeMsg },
            { no: 2, name: "edges", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => EdgeMsg }
        ]);
    }
}
/**
 * @generated MessageType for protobuf message GraphUpdated
 */
export const GraphUpdated = new GraphUpdated$Type();
/**
 * @generated ServiceType for protobuf service MyService
 */
export const MyService = new ServiceType("MyService", [
    { name: "GetCapabilities", options: {}, I: Void, O: Capabilities },
    { name: "GraphUpdate", options: {}, I: GraphUpdated, O: Void }
]);
