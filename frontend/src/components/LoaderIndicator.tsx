import { Group, Loader, Tooltip } from "@mantine/core";
import { useStore } from "@nanostores/react";
import { IconCircleCheck } from "@tabler/icons-react";
import { $isSyncing, $lastSync } from "../globalStore/loadingStore";
import { timeElapsed } from "../utils/timeUtils";

export const LoaderIndicator = () => {
  const isSyncing = useStore($isSyncing);
  const lastSync = useStore($lastSync);

  return (
    <Tooltip
      label={isSyncing ? "Syncing..." : `Saved: ${timeElapsed(lastSync)}`}
    >
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
