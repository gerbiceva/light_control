import { Group, Loader } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { IconSquareRoundedCheck } from "@tabler/icons-react";
import { $isSyncing } from "../globalStore/loadingStore";

export const LoaderIndicator = () => {
  const isSyncing = useStore($isSyncing);
  return (
    <Group align="center">
      {isSyncing ? (
        <Loader size="1.3rem" color="white" />
      ) : (
        <IconSquareRoundedCheck size="1.3rem" />
      )}
    </Group>
  );
};
