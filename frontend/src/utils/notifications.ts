import { notifications } from "@mantine/notifications";
import { theme } from "../theme";

interface NotifMessage {
  title: string;
  message: string;
}

export const notifSuccess = ({ message, title }: NotifMessage) => {
  notifications.show({
    title,
    message,
    color: theme.colors["green"][5],
  });
};

export const notifError = ({ message, title }: NotifMessage) => {
  notifications.show({
    title,
    message,
    color: theme.colors["red"][5],
  });
};

export const notifLog = ({ message, title }: NotifMessage) => {
  notifications.show({
    title,
    message,
  });
};
