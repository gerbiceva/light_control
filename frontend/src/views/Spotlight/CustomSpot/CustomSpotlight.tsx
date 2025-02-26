import {
  Badge,
  Button,
  ColorSwatch,
  Divider,
  Flex,
  Group,
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
import { filterItemsByName, filterItemsByType } from "./filtering/filter";
import { useStore } from "@nanostores/react";
import {
  $spotFilter,
  resetSpotFilter,
} from "../../../globalStore/spotlightFilterStore";
import { BaseType } from "../../../grpc/client_code/service";
import { freezeMousePos } from "../../../globalStore/mouseStore";

export interface CustomSpotlightGroups {
  data: CustomSpotData[];
  groupName: string;
}

export const CustomSpotlight = () => {
  const [query, setQuery] = useState("");
  const data = useActions();
  const spotFilter = useStore($spotFilter);

  const filter = useCallback(
    (data: CustomSpotData[]) => {
      // filter by query
      const queryFiltered = filterItemsByName(
        data,
        (item) =>
          [
            item.capability.name,
            item.capability.description,
            item.capability.namespace,
          ].join("/"),

        query,
      );

      if (!spotFilter) {
        return queryFiltered;
      }

      const dataTypeFiltered = filterItemsByType(queryFiltered, spotFilter);
      return dataTypeFiltered;
    },
    [query, spotFilter],
  );

  const items = useMemo(() => {
    return data.map((grp) => (
      <Stack key={grp.groupName} gap="sm">
        <Text fw="bold" c="dimmed" px="md" pt="xl">
          {grp.groupName}
        </Text>
        {filter(grp.data).map((item) => (
          <Spotlight.Action
            key={item.label + item.capability.namespace}
            onClick={item.onClick}
          >
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
  }, [data, filter]);

  const openSpot = useCallback(() => {
    freezeMousePos();
    resetSpotFilter();
    spotlight.open();
  }, []);

  useHotkeys([
    ["mod+F", openSpot],
    ["ctrl+space", openSpot],
  ]);

  return (
    <>
      <Button onClick={openSpot} variant="light">
        <Group>
          {/* <Kbd>CTRl + SPACE</Kbd> */}
          <Text> Node picker</Text>
        </Group>
      </Button>

      <Spotlight.Root query={query} onQueryChange={setQuery} centered>
        <Stack p="xs" gap="4px" px="md">
          <Group gap="md" wrap="nowrap">
            <IconSearch />
            <Spotlight.Search
              placeholder="Search..."
              style={{
                width: "100%",
              }}
            />
          </Group>
          {spotFilter && (
            <Group align="center" w="100%">
              <Text size="sm" c="dimmed">
                {spotFilter.type == "source" ? "Inputs:" : "Outputs:"}
              </Text>

              <Badge
                variant="dot"
                size="lg"
                color={getColorFromEnum(spotFilter.dataType)[5]}
              >
                {BaseType[spotFilter.dataType]}
              </Badge>
            </Group>
          )}
        </Stack>
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
