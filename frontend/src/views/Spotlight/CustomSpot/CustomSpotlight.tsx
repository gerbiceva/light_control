import {
  Badge,
  Button,
  ColorSwatch,
  Divider,
  Flex,
  Group,
  Kbd,
  Stack,
  Text,
} from "@mantine/core";
import { useHotkeys } from "@mantine/hooks";
import { Spotlight, spotlight } from "@mantine/spotlight";
import { IconSearch } from "@tabler/icons-react";
import { useCallback, useMemo, useState } from "react";
import {
  getColorFromEnum,
  getColorFromString,
} from "../../../utils/colorUtils";
import { useActions } from "../UseActions";
import { CustomSpotData } from "./CustomSpotData";

export interface CustomSpotlightGroups {
  data: CustomSpotData[];
  groupName: string;
}

export const CustomSpotlight = () => {
  const [query, setQuery] = useState("");

  const data = useActions();

  const isSearchFoundInclude = useCallback(
    (item: CustomSpotData) => {
      const searchStr = (item.label + "" + item.description)
        .toLowerCase()
        .trim();
      return searchStr.includes(query.toLowerCase().trim());
    },
    [query]
  );

  const items = useMemo(() => {
    return data.map((grp) => (
      <Stack key={grp.groupName} gap="sm">
        <Text fw="bold" c="dimmed" px="md" pt="xl">
          {grp.groupName}
        </Text>
        {grp.data
          .filter((item) => isSearchFoundInclude(item))
          .map((item) => (
            <Spotlight.Action key={item.label} onClick={item.onClick}>
              <Group wrap="nowrap" w="100%" align="center">
                {item.leftSection}
                <Stack gap="6px" w="100%">
                  <Flex
                    gap="sm"
                    align="center"
                    w="100%"
                    wrap="nowrap"
                    justify="space-between"
                  >
                    <Text size="md" style={{ textWrap: "nowrap" }}>
                      {item.label}
                    </Text>
                    <Divider opacity={0.6} style={{ flex: 1 }} />
                    <Badge
                      variant="light"
                      size="xs"
                      color={getColorFromString(item.capability.namespace)[5]}
                    >
                      {item.capability.namespace}
                    </Badge>
                  </Flex>
                  <Group
                    align="center"
                    w="100%"
                    justify="space-between"
                    wrap="nowrap"
                    gap="xl"
                  >
                    <Text size="xs" c="dimmed">
                      {item.description}
                    </Text>

                    {grp.groupName != "Inputs" && (
                      <Group wrap="nowrap" gap="md" justify="end">
                        <Group gap="xs" wrap="nowrap">
                          {item.capability.inputs.map((inp) => (
                            <ColorSwatch
                              key={inp.name}
                              size="14px"
                              color={getColorFromEnum(inp.type)[5]}
                            />
                          ))}
                        </Group>
                        <Divider orientation="vertical" />
                        <Group gap="xs" wrap="nowrap">
                          {item.capability.outputs.map((inp) => (
                            <ColorSwatch
                              key={inp.name}
                              size="14px"
                              color={getColorFromEnum(inp.type)[5]}
                            />
                          ))}
                        </Group>
                      </Group>
                    )}
                  </Group>
                </Stack>
              </Group>
            </Spotlight.Action>
          ))}
      </Stack>
    ));
  }, [data, isSearchFoundInclude]);

  useHotkeys([
    ["mod+F", spotlight.open],
    ["ctrl+space", spotlight.open],
  ]);

  return (
    <>
      <Button onClick={spotlight.open} variant="subtle">
        <Group>
          <Kbd>CTRl + SPACE</Kbd>
          <Text c="dimmed"> open node picker</Text>
        </Group>
      </Button>

      <Spotlight.Root query={query} onQueryChange={setQuery} centered>
        {/* <Card p="0px" withBorder={isRegex}> */}
        <Group gap="md" wrap="nowrap" p="sm" px="md">
          <IconSearch />
          <Spotlight.Search
            placeholder="Search..."
            style={{
              width: "100%",
            }}
          />
        </Group>
        {/* </Card> */}
        <Spotlight.ActionsList mah="80vh">
          {items.length > 0 ? (
            items
          ) : (
            <Spotlight.Empty>Nothing found...</Spotlight.Empty>
          )}
        </Spotlight.ActionsList>
      </Spotlight.Root>
    </>
  );
};
