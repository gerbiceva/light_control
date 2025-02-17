import {
  Card,
  Group,
  Loader,
  LoadingOverlay,
  SimpleGrid,
  Stack,
  TextInput,
  Title,
  Text,
} from "@mantine/core";
import { NodeView } from "./NodeView";
import { useStore } from "@nanostores/react";
import { $serverCapabilities } from "../globalStore/capabilitiesStore";
import { LoaderIndicator } from "../components/LoaderIndicator";
import { CustomSpotlight } from "./Spotlight/CustomSpot/CustomSpotlight";
import { SettingsModal } from "../components/settingsModal/SettingsModal";
import { $projectName, setProjectName } from "../globalStore/projectStore";
import { BlindSwitch } from "../components/BlindSwitch";

export const MainLayout = () => {
  const caps = useStore($serverCapabilities);
  const projectName = useStore($projectName);

  if (caps.length == 0) {
    return (
      <LoadingOverlay
        visible={true}
        loaderProps={{
          children: (
            <Stack align="center" gap="xl">
              <Loader />
              <Title size="sm">Loading node capabilities from the server</Title>
            </Stack>
          ),
        }}
      />
    );
  }

  return (
    <Stack w="100vw" h="100vh" p="sm" gap="sm">
      <Card withBorder shadow="md">
        <SimpleGrid cols={3} px="lg">
          {/* title */}
          <Group w="100%">
            <img src="/icon.svg" height="30px" width="30px"></img>
            {/* <Title size="lg">LightControll</Title> */}
            <Group gap="0">
              <TextInput
                fw="bold"
                variant="filled"
                value={projectName}
                onChange={(ev) => {
                  setProjectName(ev.target.value);
                }}
              />
              <Text c="dimmed" size="xs" fw="bold">
                .json
              </Text>
            </Group>
          </Group>
          {/* spotlight for adding nodes */}
          <CustomSpotlight />
          {/*Settings and server indicator*/}
          <Group gap="lg" justify="end">
            <BlindSwitch />
            <LoaderIndicator />
            <SettingsModal />
          </Group>
        </SimpleGrid>
      </Card>

      <NodeView />
    </Stack>
  );
};
