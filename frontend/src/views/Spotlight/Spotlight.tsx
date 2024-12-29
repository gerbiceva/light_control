import { rem, Button, Kbd, Text, Group } from "@mantine/core";
import { Spotlight, spotlight } from "@mantine/spotlight";
import { IconSearch } from "@tabler/icons-react";
import { theme } from "../../theme";
import { UseActions } from "./UseActions";

export const NodeAdderSpotlight = () => {
  const actions = UseActions();
  return (
    <>
      <Button onClick={spotlight.open} variant="subtle">
        <Group>
          <Kbd>CTRl + K</Kbd>
          <Text c="dimmed"> open node picker</Text>
        </Group>
      </Button>
      {theme && (
        <Spotlight
          color="gray"
          actions={actions}
          nothingFound="Nothing found..."
          highlightQuery
          scrollable
          maxHeight={400}
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
      )}
    </>
  );
};
