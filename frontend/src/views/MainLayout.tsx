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
  Box,
  Divider,
  ActionIcon,
  Drawer,
  Button,
} from "@mantine/core";
import { NodeView } from "./NodeView";
import { useStore } from "@nanostores/react";
import { $serverCapabilities } from "../globalStore/capabilitiesStore";
import { LoaderIndicator } from "../components/LoaderIndicator";
import { CustomSpotlight } from "./Spotlight/CustomSpot/CustomSpotlight";
import { SettingsModal } from "../components/settingsModal/SettingsModal";
import { $projectName } from "../globalStore/projectStore";
import { BlindSwitch } from "../components/BlindSwitch";
import { SubgraphTab } from "../components/Subgraph/Tabs/SubgraphTab";
import { IconSettings2 } from "@tabler/icons-react";

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
    <Stack w="100vw" h="100vh" p="sm" gap="sm" pos="relative">
      <Card withBorder shadow="md">
        <SimpleGrid cols={3} px="lg">
          {/* title */}
          <Group w="100%">
            <img src="/icon.svg" height="30px" width="30px"></img>
            {/* <Title size="lg">LightControll</Title> */}

            <Text>{projectName}</Text>
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

      <Drawer
        opened={false}
        title="Subgraphs"
        onClose={function (): void {
          throw new Error("Function not implemented.");
        }}
      >
        asdas
      </Drawer>

      <Group pos="absolute" top="4.5rem" w="100%" m="0" p="0">
        <Card
          withBorder
          shadow="xl"
          p="xs"
          style={{
            borderTop: "none",
            borderTopLeftRadius: 0,
            borderTopRightRadius: 0,
            zIndex: 2,
          }}
        >
          <Group gap="0" w="100%">
            <Button leftSection={<IconSettings2 />} size="xs">
              Subgraphs
            </Button>
            {Array(4)
              .fill(null)
              .map((_, i) => (
                <>
                  <SubgraphTab
                    key={i}
                    name={`Subgraph ${i + 1}  `}
                    onClose={function (): void {
                      throw new Error("Function not implemented.");
                    }}
                  />
                  {/* <Divider orientation="vertical" /> */}
                </>
              ))}
          </Group>
        </Card>

        <Text px="md" size="md" fw="bold">
          Ime grafa
        </Text>
      </Group>

      <NodeView />
    </Stack>
  );
};
