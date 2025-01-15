import {
  ActionIcon,
  Card,
  Group,
  Loader,
  LoadingOverlay,
  SimpleGrid,
  Stack,
  Title,
} from "@mantine/core";
import { NodeView } from "./NodeView";
import { useStore } from "@nanostores/react";
import { $serverCapabilities } from "../globalStore/capabilitiesStore";
import { LoaderIndicator } from "../components/LoaderIndicator";
import { CustomSpotlight } from "./Spotlight/CustomSpot/CustomSpotlight";
import { IconRecycle, IconTrash } from "@tabler/icons-react";
import { resetState } from "../globalStore/flowStore";

export const MainLayout = () => {
  const caps = useStore($serverCapabilities);

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
          <Group>
            <img src="/icon.svg" height="30px" width="30px"></img>
            <Title size="lg">LightControll</Title>
          </Group>
          {/* spotlight for adding nodes */}
          <CustomSpotlight />
          {/*Settings and server indicator*/}
          <Group gap="lg" justify="end">
            <LoaderIndicator />
            <ActionIcon
              variant="subtle"
              onClick={() => {
                if (
                  !confirm("reset nodes and edges? Operation can't be undone")
                ) {
                  return;
                }

                resetState();
              }}
            >
              <IconRecycle />
            </ActionIcon>
          </Group>
        </SimpleGrid>
      </Card>

      <NodeView />
    </Stack>
  );
};
