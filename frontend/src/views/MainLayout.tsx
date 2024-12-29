import {
  ActionIcon,
  Button,
  Card,
  Group,
  Indicator,
  LoadingOverlay,
  Stack,
  Title,
  Tooltip,
} from "@mantine/core";
import { NodeAdderSpotlight } from "./Spotlight/Spotlight";
import { NodeView } from "./NodeView";
import { IconSettings } from "@tabler/icons-react";
import { useStore } from "@nanostores/react";
import { $capabilities } from "../globalStore/capabilitiesStore";

export const MainLayout = () => {
  const caps = useStore($capabilities);

  if (caps.length == 0) {
    return <LoadingOverlay visible={true} />;
  }

  return (
    <Stack w="100vw" h="100vh" p="lg" gap="xl">
      <Card withBorder shadow="md">
        <Group justify="space-between" px="lg">
          {/* title */}
          <Title size="lg">LightControll</Title>
          {/* spotlight for adding nodes */}
          <NodeAdderSpotlight />
          {/*Settings and server indicator*/}
          <Group gap="xl">
            <Tooltip label="Connection to server is active. Click to force apply layout">
              <Indicator processing color="green" size="1rem">
                <Button
                  size="sm"
                  onClick={async () => {
                    // const data = await client.({ name: "lan" });
                    // console.log(data.response.message);
                  }}
                >
                  Force apply
                </Button>
              </Indicator>
            </Tooltip>
            <ActionIcon variant="light">
              <IconSettings />
            </ActionIcon>
          </Group>
        </Group>
      </Card>

      <NodeView />
    </Stack>
  );
};
