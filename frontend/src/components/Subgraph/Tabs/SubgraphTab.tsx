import {
  ActionIcon,
  Button,
  Group,
  Text,
  useMantineTheme,
} from "@mantine/core";
import { IconX } from "@tabler/icons-react";
import { setActiveSubgraph } from "../../../globalStore/flowStore";
import { SubGraph } from "../../../subgraph/Subgraph";

interface SubgraphTabProps {
  onClose: () => void;
  active: boolean;
  subgraph: SubGraph;
}

export const SubgraphTab = ({
  subgraph,
  onClose,
  active,
}: SubgraphTabProps) => {
  // a tab with name, onclick handler and a close icon with x
  const theme = useMantineTheme();
  return (
    <Group
      gap="-20px"
      bg={
        active
          ? // @ts-expect-error from colo
            theme.colors[theme.primaryColor][theme.primaryShade.light]
          : theme.colors.gray[0]
      }
    >
      <Button
        variant={active ? "filled" : "subtle"}
        size="xs"
        onClick={() => {
          setActiveSubgraph(subgraph.id);
        }}
      >
        <Text size="xs">{subgraph.name}</Text>
      </Button>
      <Group p="xs" ml="-8px">
        <ActionIcon
          onClick={onClose}
          size="xs"
          variant="subtle"
          disabled={active}
          opacity={active ? 0 : 1}
        >
          <IconX opacity={0.2} size={14} />
        </ActionIcon>
      </Group>
    </Group>
  );
};
