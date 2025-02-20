import { Flex, LoadingOverlay, SimpleGrid, Stack, Title } from "@mantine/core";
import { Slider } from "../components/TouchComponents/Slider";

const cols = 10;
// url of window.location.host
const wsUrl = `ws://${window.location.host}/ws`;

export const TouchUI = () => {
  // const { connected, start, stop, sendWsData } = useWebsocket(wsUrl, (data) => {
  //   console.log(data);
  // });

  // // send websocket data using fader index and value from 0 to 100
  // const sendFaderData = useCallback(
  //   (index: number, value: number) => {
  //     // first byte is the fader index, second byte is the value
  //     const data = new Uint8Array([index, value]);
  //     sendWsData(data);
  //   },
  //   [sendWsData],
  // );

  // useEffect(() => {
  //   if (!connected) {
  //     start();
  //   }

  //   return () => {
  //     stop();
  //   };
  // }, [start, stop]);

  // if (!connected) {
  //   return <LoadingOverlay visible={true} />;
  // }

  return (
    <Stack h="100%" p="xl">
      <Title>Touch UI</Title>
      <SimpleGrid h="100%" cols={cols}>
        {Array(cols * 2)
          .fill(null)
          .map((_, index) => (
            <Flex align="center" justify="center" key={index}>
              <Slider
                baseWidth="80%"
                // onChange={(data) => {
                //   sendFaderData(index, data);
                // }}
              />
            </Flex>
          ))}
      </SimpleGrid>
    </Stack>
  );
};
