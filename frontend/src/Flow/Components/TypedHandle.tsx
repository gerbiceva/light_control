import { Box, ColorSwatch } from "@mantine/core";
import { Handle, Position } from "@xyflow/react";

interface TypedHandleProps {
  color: string;
  id: string;
}

export const TypedHandle = ({ color, id }: TypedHandleProps) => {
  return (
    <Box>
      <ColorSwatch
        color={color}
        style={{
          transform: "translate(15%, 50%)",
        }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id={id}
        style={{
          opacity: 0,
          margin: "0",
          padding: "0",
          left: "0",
          bottom: "0",
          position: "relative",
          width: "2rem",
          height: "2rem",
          borderRadius: "0",
          border: "none",
        }}
      />
    </Box>
  );
};
