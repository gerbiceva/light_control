import { Handle, HandleType, Position } from "@xyflow/react";

interface TypedHandleProps {
  color: string;
  id: string;
  pos?: Position;
  type?: HandleType;
}

export const TypedHandle = ({
  color,
  id,
  type = "source",
}: TypedHandleProps) => {
  return (
    <Handle
      type={type}
      position={type == "source" ? Position.Right : Position.Left}
      id={id}
      style={{
        backgroundColor: color,
        transform: "none",
        left: "0",
        bottom: "0",
        top: 0,
        right: 0,
        borderRadius: 0,
        border: "none",
        position: "relative",
        width: "1.4rem",
        height: "1.4rem",
      }}
    />
  );
};
