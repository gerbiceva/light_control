import { Flex, SimpleGrid } from "@mantine/core";
import { Slider } from "../components/TouchComponents/Slider";

const cols = 8;

export const TouchUI = () => {
  return (
    <SimpleGrid cols={cols} h="100%" p="md">
      {Array(cols * 2)
        .fill(null)
        .map((_, index) => (
          <Flex align="center" justify="center" key={index}>
            <Slider baseWidth={90} />
          </Flex>
        ))}
    </SimpleGrid>
  );
};
