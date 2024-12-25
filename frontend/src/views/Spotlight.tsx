import { rem, Button, Kbd, Text, Group } from "@mantine/core";
import { Spotlight, SpotlightActionData, spotlight } from "@mantine/spotlight";
import { IconSearch, IconGraph, IconNumber1 } from "@tabler/icons-react";

const actions: SpotlightActionData[] = [
  {
    id: "Int",
    label: "Int",
    description: "Whole number input",
    onClick: () => console.log("Home"),
    leftSection: (
      <IconNumber1 style={{ width: rem(24), height: rem(24) }} stroke={1.5} />
    ),
  },
  {
    id: "curve",
    label: "Curve",
    description: "Parametric curve input",
    onClick: () => console.log("Dashboard"),
    leftSection: (
      <IconGraph style={{ width: rem(24), height: rem(24) }} stroke={1.5} />
    ),
  },
];

export const NodeAdderSpotlight = () => {
  return (
    <>
      <Button onClick={spotlight.open} variant="subtle">
        <Group>
          <Kbd>CTRl + K</Kbd>
          <Text c="dimmed"> open node picker</Text>
        </Group>
      </Button>
      <Spotlight
        actions={actions}
        nothingFound="Nothing found..."
        highlightQuery
        searchProps={{
          leftSection: (
            <IconSearch
              style={{ width: rem(20), height: rem(20) }}
              stroke={1.5}
            />
          ),
          placeholder: "Search...",
        }}
      />
    </>
  );
};
