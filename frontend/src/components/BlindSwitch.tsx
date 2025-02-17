import { Tooltip, Switch } from "@mantine/core";
import { IconRefresh, IconEyeOff } from "@tabler/icons-react";
import { $sync, setAutoUpdate } from "../globalStore/loadingStore";
import { useStore } from "@nanostores/react";

export const BlindSwitch = () => {
  const { autoUpdate } = useStore($sync);

  const toggleBlindMode = () => {
    setAutoUpdate(!autoUpdate);
  };

  return (
    <Tooltip label="Toggle BLIND mode. Enable/disable automatic updates.">
      <div>
        <Switch
          size="md"
          color="dark.4"
          onLabel={<IconRefresh size={16} stroke={2.5} />}
          offLabel={<IconEyeOff size={16} stroke={2.5} />}
          checked={autoUpdate}
          onChange={toggleBlindMode}
        />
      </div>
    </Tooltip>
  );
};
