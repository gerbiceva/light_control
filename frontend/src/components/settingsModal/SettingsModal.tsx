import {
  ActionIcon,
  Button,
  Divider,
  Modal,
  SimpleGrid,
  Stack,
  Text,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconDownload, IconRecycle, IconSettings } from "@tabler/icons-react";
import { $nodes, resetState } from "../../globalStore/flowStore";
import { SettingsDrop } from "./SettingsDrop";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { downloadSettings } from "./fileUtils";

export const SettingsModal = () => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <>
      <Modal opened={opened} onClose={close} title="Settings" centered>
        <Stack>
          <Divider label="Graph settings" labelPosition="left" />
          <Button
            variant="light"
            onClick={downloadSettings}
            leftSection={<IconDownload />}
          >
            Save file
          </Button>
          {/* clear */}
          <Button
            variant="light"
            color="red"
            onClick={() => {
              if (
                !confirm("reset nodes and edges? Operation can't be undone")
              ) {
                return;
              }

              resetState();
            }}
            leftSection={<IconRecycle />}
          >
            Reset graph to defaults
          </Button>
          <SettingsDrop />

          <Divider label="Stats" labelPosition="left" />
          <SimpleGrid cols={2} spacing="sm">
            <Text c="dimmed">Server address</Text>
            <Text>{window.location.hostname}:50051</Text>
            <Text c="dimmed">Capabilities loaded</Text>
            <Text>{$capabilities.get().length}</Text>
            <Text c="dimmed">Node count</Text>
            <Text>{$nodes.get().length}</Text>
          </SimpleGrid>
        </Stack>
      </Modal>

      <ActionIcon variant="subtle" onClick={open}>
        <IconSettings />
      </ActionIcon>
    </>
  );
};
