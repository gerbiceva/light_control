import { Stack, Text } from "@mantine/core";
import { IconUpload, IconX, IconJson } from "@tabler/icons-react";
import { Dropzone, DropzoneProps } from "@mantine/dropzone";
// import { readFile } from "./fileUtils";
// import { $edges, $nodes } from "../../globalStore/flowStore";
// import { notifError, notifSuccess } from "../../utils/notifications";

export const SettingsDrop = (props: Partial<DropzoneProps>) => {
  return (
    <Dropzone
      onDrop={() => {
        // readFile<>(files[0])
        //   .then((f) => {
        //     $edges.set(f.edges);
        //     $nodes.set(f.nodes);
        //     notifSuccess({
        //       title: "Graph loaded",
        //       message: "all settings have been restored",
        //     });
        //   })
        //   .catch((reason) => {
        //     notifError({
        //       title: "Cant load settings",
        //       message: reason,
        //     });
        //   });
      }}
      onReject={(files) => console.log("rejected files", files)}
      accept={["application/json"]}
      {...props}
    >
      <Stack
        align="center"
        justify="center"
        gap="xl"
        mih={220}
        style={{ pointerEvents: "none" }}
      >
        <Dropzone.Accept>
          <IconUpload size={52} stroke={1.5} />
        </Dropzone.Accept>
        <Dropzone.Reject>
          <IconX size={52} color="var(--mantine-color-red-6)" stroke={1.5} />
        </Dropzone.Reject>
        <Dropzone.Idle>
          <IconJson
            size={52}
            color="var(--mantine-color-dimmed)"
            stroke={1.5}
          />
        </Dropzone.Idle>

        <div>
          <Text size="md" inline>
            Drag saved graph file here
          </Text>
          <Text size="xs" c="dimmed" inline mt={7}>
            your settings will be imported.
          </Text>
        </div>
      </Stack>
    </Dropzone>
  );
};
