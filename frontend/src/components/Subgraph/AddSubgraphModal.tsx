import {
  Button,
  Divider,
  Modal,
  Stack,
  Textarea,
  TextInput,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { SubGraph } from "./Subgraph";
import { useForm } from "@mantine/form";
import { useSubgraphs } from "../../globalStore/subgraphStore";

export const AddSubgraphModal = () => {
  const [opened, { open, close }] = useDisclosure(false);
  const { newSubGraph } = useSubgraphs();

  const form = useForm<Pick<SubGraph, "description" | "name">>({
    mode: "uncontrolled",
    initialValues: {
      name: "",
      description: undefined,
    },

    validate: {
      name: (value) =>
        value.length > 3 ? null : "Name must be atleast 4 letters long",
    },
  });

  return (
    <>
      <Modal opened={opened} onClose={close} title="Add subgraph" centered>
        <form
          onSubmit={form.onSubmit((values) => {
            console.log(values);
            newSubGraph(values.name, values.description);
            close();
            return;
          })}
        >
          <Stack>
            <Divider label="Subgraph settings" labelPosition="left" />

            <TextInput
              autoComplete="off"
              description="Graph name"
              fw="bold"
              variant="filled"
              {...form.getInputProps("name")}
              error={form.errors && form.errors.name}
            />
            <Textarea
              autoComplete="off"
              description="Description"
              variant="filled"
              placeholder="Controlls intensitiy with audio..."
              {...form.getInputProps("description")}
            />

            <Button mt="lg" type="submit">
              Add
            </Button>
          </Stack>
        </form>
      </Modal>
      <Button fullWidth my="xl" onClick={open}>
        Add subgraph
      </Button>
    </>
  );
};
