import { Group, Loader, Tooltip } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { IconCircle, IconCircleCheck } from "@tabler/icons-react";
import { $isSyncing, $sync } from "../globalStore/loadingStore";
import { useLastSync } from "../utils/useLastSync";

export const LoaderIndicator = () => {
  const isSyncing = useStore($isSyncing);
  const preetyTimeElapsed = useLastSync();
  const { autoUpdate } = useStore($sync);

  return (
    <Tooltip label={isSyncing ? "Syncing..." : `Saved: ${preetyTimeElapsed}`}>
      <Group align="center" opacity={0.4}>
        {autoUpdate ? (
          isSyncing ? (
            <Loader size="1.4rem" m="0.2rem" />
          ) : (
            <IconCircleCheck size="1.8rem" />
          )
        ) : (
          <IconCircle size="1.8rem" />
        )}
      </Group>
    </Tooltip>
  );
};
