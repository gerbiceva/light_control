import { Group, Loader, Tooltip } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { IconCircleCheck } from "@tabler/icons-react";
import { $isSyncing } from "../globalStore/loadingStore";
import { useLastSync } from "../utils/useLastSync";

export const LoaderIndicator = () => {
  const isSyncing = useStore($isSyncing);
  const preetyTimeElapsed = useLastSync();

  return (
    <Tooltip label={isSyncing ? "Syncing..." : `Saved: ${preetyTimeElapsed}`}>
      <Group align="center" opacity={0.4}>
        {isSyncing ? (
          <Loader size="1.4rem" m="0.2rem" />
        ) : (
          <IconCircleCheck size="1.8rem" />
        )}
      </Group>
    </Tooltip>
  );
};
