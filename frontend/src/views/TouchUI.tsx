import {
  Card,
  Flex,
  Group,
  LoadingOverlay,
  SimpleGrid,
  Stack,
  Title,
  Text,
} from "@mantine/core";
import { Slider } from "../components/TouchComponents/Slider";
import { useWebsocket } from "../ws/websocketHandler";
import { useCallback } from "react";
import { IconPlugConnected, IconPlugConnectedX } from "@tabler/icons-react";
import { theme } from "../theme";

const cols = 10;
// url of window.location.host
const wsUrl = `ws://${window.location.hostname}:8080/ws`;

export const TouchUI = () => {
  const { connected, sendWsData } = useWebsocket(wsUrl, (data) => {
    console.log(data);
  });
  console.log("touch ui updated");

  // send websocket data using fader index and value from 0 to 100
  const sendFaderData = useCallback(
    (index: number, value: number) => {
      // first byte is the fader index, second byte is the value
      const data = new Uint8Array([index, value]);
      sendWsData(data);
    },
    [sendWsData]
  );

  if (!connected) {
    return <LoadingOverlay visible={true} />;
  }

  return (
    <Stack h="100%" p="xs">
      <Card withBorder shadow="md">
        <SimpleGrid cols={3} px="lg">
          {/* title */}

          <Group w="100%">
            <img src="/icon.svg" height="30px" width="30px"></img>
            <Title size="lg">Touch UI</Title>
          </Group>
          <div></div>
          <Group w="100%" justify="end">
            {connected ? (
              <Group>
                <IconPlugConnected size={24} color={theme.colors["green"][5]} />
                <Text size="xs">Link connected</Text>
              </Group>
            ) : (
              <Group>
                <IconPlugConnectedX size={24} color={theme.colors["red"][5]} />
                <Text size="xs">Link disconnected</Text>
              </Group>
            )}
          </Group>
        </SimpleGrid>
      </Card>
      <SimpleGrid h="100%" cols={cols} pb="5rem">
        {Array(cols * 2)
          .fill(null)
          .map((_, index) => (
            <Flex align="center" justify="center" key={index} py="4rem" px="md">
              <Slider
                baseWidth="80%"
                onChange={(data) => {
                  sendFaderData(index, data);
                }}
              />
            </Flex>
          ))}
      </SimpleGrid>
    </Stack>
  );
};
