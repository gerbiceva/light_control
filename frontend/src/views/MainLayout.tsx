import {
  Card,
  Group,
  Loader,
  LoadingOverlay,
  Stack,
  Title,
} from "@mantine/core";
import { NodeAdderSpotlight } from "./Spotlight/Spotlight";
import { NodeView } from "./NodeView";
import { useStore } from "@nanostores/react";
import { $serverCapabilities } from "../globalStore/capabilitiesStore";
import { LoaderIndicator } from "../components/LoaderIndicator";

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
        <Group justify="space-between" px="lg">
          {/* title */}
          <Title size="lg">LightControll</Title>
          {/* spotlight for adding nodes */}
          <NodeAdderSpotlight />
          {/*Settings and server indicator*/}
          <Group gap="lg">
            <LoaderIndicator />

            {/* <Tooltip label="Connection to server is active.">
              <Indicator
                processing
                color="green"
                size="0.7rem"
                position="top-end"
              >
                <div />
              </Indicator>
            </Tooltip> */}
          </Group>
        </Group>
      </Card>

      <NodeView />
    </Stack>
  );
};
