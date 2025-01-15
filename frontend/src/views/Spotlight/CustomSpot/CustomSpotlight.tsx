import { useCallback, useMemo, useState } from "react";
import { Spotlight, spotlight } from "@mantine/spotlight";
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
import { IconSearch } from "@tabler/icons-react";
import { CustomSpotData } from "./CustomSpotData";
import {
  getColorFromEnum,
  getColorFromString,
} from "../../../utils/colorUtils";
import { useHotkeys } from "@mantine/hooks";
import { useActions } from "../UseActions";

export interface CustomSpotlightGroups {
  data: CustomSpotData[];
  groupName: string;
}

export const CustomSpotlight = () => {
  const [query, setQuery] = useState("");
  const data = useActions();

  const isSearchFound = useCallback(
    (item: CustomSpotData) => {
      const searchStr = (item.name + "" + item.description).toLowerCase();
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
          .filter((item) => isSearchFound(item))
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
  }, [data, isSearchFound]);

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
        <Spotlight.Search
          placeholder="Search..."
          leftSection={<IconSearch stroke={1.5} />}
        />
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
