import { Group, Loader } from "@mantine/core";
import { IconSquareRoundedCheck } from "@tabler/icons-react";

export interface LoaderIndicatorProps {
  isLoading: boolean;
}
export const LoaderIndicator = ({ isLoading }: LoaderIndicatorProps) => {
  return (
    <Group align="center">
      {isLoading ? (
        <Loader size="1.3rem" color="white" />
      ) : (
        <IconSquareRoundedCheck size="1.3rem" />
      )}
    </Group>
  );
};
