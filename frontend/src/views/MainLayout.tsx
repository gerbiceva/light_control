import {
  Card,
  Divider,
  Group,
  Loader,
  LoadingOverlay,
  SimpleGrid,
  Stack,
  Title,
  Text,
  Badge,
  Box,
  Button,
  Drawer,
} from "@mantine/core";
import { LoaderIndicator } from "../components/LoaderIndicator";
import { SettingsModal } from "../components/settingsModal/SettingsModal";
import { NodeView } from "./GraphView";
import { CustomSpotlight } from "./Spotlight/CustomSpot/CustomSpotlight";

import { ReactFlowProvider } from "@xyflow/react";
import { BlindSwitch } from "../components/BlindSwitch";
// import { $syncedAppState } from "../crdt/repo";
import { $syncedAppState, useYjsState } from "../crdt/globalSync";
import { useDisclosure } from "@mantine/hooks";
import { useStore } from "@nanostores/react";
import { $serverCapabilities } from "../globalStore/capabilitiesStore";
import { addVisibleSubgraph, useSubgraphs } from "../globalStore/subgraphStore";
import { IconEdit, IconSettings2 } from "@tabler/icons-react";
import { AddSubgraphModal } from "../components/Subgraph/AddSubgraphModal";
import { SubgraphTab } from "../components/Subgraph/Tabs/SubgraphTab";

export const MainLayout = () => {
  const caps = useStore($serverCapabilities);
  const appState = useStore($syncedAppState);
  const { activeGraph, visibleGraphs, setActiveGraph } = useSubgraphs();
  const { isLoading } = useYjsState();

  const [opened, { close, toggle }] = useDisclosure(false);
  if (caps.length == 0 || appState == undefined || isLoading) {
    return (
      <LoadingOverlay
        visible
        loaderProps={{
          children: (
            <Stack align="center" gap="xl">
              <Loader />
              <Title size="sm">Loading node capabilities, and app state</Title>
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

            <Text>{activeGraph?.name}</Text>
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

      <Drawer opened={opened} title="Subgraphs" onClose={close}>
        <Stack>
          {[appState.main]
            .concat(Object.values(appState.subgraphs))
            .map((graph) => (
              <Card key={graph.id} withBorder shadow="xl">
                <Stack>
                  <Group w="100%" justify="space-between">
                    <Box>
                      <Badge size="sm" variant="light">
                        {graph.id}
                      </Badge>
                      <Title order={2}>{graph.name}</Title>
                    </Box>
                    <Button
                      variant="light"
                      rightSection={<IconEdit size={16} />}
                      onClick={() => {
                        addVisibleSubgraph(graph.id);
                        close();
                      }}
                    >
                      edit
                    </Button>
                  </Group>
                  <Text>{graph.description}</Text>
                  <Group justify="space-around">
                    <Stack gap="xs">
                      <Text c="dimmed" size="xs" fw="bold">
                        NODES:
                      </Text>
                      <Text size="md">{graph.nodes.length}</Text>
                    </Stack>
                    <Stack gap="xs">
                      <Text c="dimmed" size="xs" fw="bold">
                        EDGES:
                      </Text>
                      <Text size="md">{graph.edges.length}</Text>
                    </Stack>
                  </Group>
                </Stack>
              </Card>
            ))}
          <AddSubgraphModal />
        </Stack>
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
          <Group gap="sm" w="100%">
            <Button
              leftSection={<IconSettings2 />}
              size="xs"
              variant="light"
              onClick={toggle}
            >
              Manage
            </Button>
            <Divider orientation="vertical" size="sm" mx="sm" />
            {visibleGraphs.concat([appState.main]).map((graph) => {
              return (
                <SubgraphTab
                  onClick={() => {
                    if (graph.name == "main") {
                      setActiveGraph("main");
                    } else {
                      setActiveGraph(graph.id);
                    }
                  }}
                  active={activeGraph?.id == graph.id}
                  key={graph.id}
                  subgraph={graph}
                  onClose={function (): void {
                    throw new Error("Function not implemented.");
                  }}
                />
              );
            })}
          </Group>
        </Card>
      </Group>

      <ReactFlowProvider>
        <NodeView />
      </ReactFlowProvider>
    </Stack>
  );
};
