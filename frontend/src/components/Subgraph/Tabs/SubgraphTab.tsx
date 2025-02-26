import {
  ActionIcon,
  alpha,
  Box,
  Button,
  Card,
  Group,
  Text,
  UnstyledButton,
  useMantineTheme,
} from "@mantine/core";
import { IconX } from "@tabler/icons-react";

interface SubgraphTabProps {
  name: string;
  onClose: () => void;
}

export const SubgraphTab = ({ name, onClose }: SubgraphTabProps) => {
  // a tab with name, onclick handler and a close icon with x
  const theme = useMantineTheme();

  return (
    <>
      <Button variant="subtle" size="xs">
        <Text size="xs">{name}</Text>
      </Button>
      <Group p="xs" px="sm">
        <ActionIcon onClick={onClose} size="xs" variant="subtle">
          <IconX opacity={0.2} size={14} />
        </ActionIcon>
      </Group>
    </>
  );
};
