import {
  ActionIcon,
  Button,
  Divider,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconDownload, IconSettings } from "@tabler/icons-react";
// import { resetState } from "../../globalStore/flowStore";
import { SettingsDrop } from "./SettingsDrop";
import { $capabilities } from "../../globalStore/capabilitiesStore";
import { downloadSettings } from "./fileUtils";
import { useStore } from "@nanostores/react";
import { $projectName, setProjectName } from "../../globalStore/projectStore";

export const SettingsModal = () => {
  const [opened, { open, close }] = useDisclosure(false);
  const projectName = useStore($projectName);
  // const appState = useStore($syncedAppState);

  return (
    <>
      <Modal opened={opened} onClose={close} title="Settings" centered>
        <Stack>
          <Divider label="Project settings" labelPosition="left" />

          <TextInput
            description="Project Name"
            fw="bold"
            variant="filled"
            value={projectName}
            onChange={(ev) => {
              setProjectName(ev.target.value);
            }}
          />

          <Divider label="Graph settings" labelPosition="left" />
          <Button
            variant="light"
            onClick={downloadSettings}
            leftSection={<IconDownload />}
          >
            Save file
          </Button>
          {/* clear */}
          {/* <Button
            variant="light"
            color="red"
            onClick={() => {
              if (
                !confirm("reset nodes and edges? Operation can't be undone")
              ) {
                return;
              }

              // resetState();
            }}
            leftSection={<IconRecycle />}
          >
            Reset graph to defaults
          </Button> */}
          <SettingsDrop />

          <Divider label="Stats" labelPosition="left" />
          <SimpleGrid cols={2} spacing="sm">
            <Text c="dimmed">Server address</Text>
            <Text>{window.location.hostname}:50051</Text>
            <Text c="dimmed">Capabilities loaded</Text>
            <Text>{$capabilities.get().length}</Text>
            {/* <Text c="dimmed">Node count</Text>
            <Text>{appState.main.nodes.length}</Text> */}
          </SimpleGrid>
        </Stack>
      </Modal>

      <ActionIcon variant="subtle" onClick={open}>
        <IconSettings />
      </ActionIcon>
    </>
  );
};
