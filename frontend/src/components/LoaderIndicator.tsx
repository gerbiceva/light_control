import { Group, Loader } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { IconSquareRoundedCheck } from "@tabler/icons-react";
import { $isSyncing } from "../globalStore/loadingStore";

export const LoaderIndicator = () => {
  const isSyncing = useStore($isSyncing);
  return (
    <Group align="center" opacity={0.4}>
      {isSyncing ? (
        <Loader size="1.8rem" />
      ) : (
        <IconSquareRoundedCheck size="1.8rem" />
      )}
    </Group>
  );
};
