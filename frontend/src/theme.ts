import { createTheme, mergeMantineTheme, DEFAULT_THEME } from "@mantine/core";

export const appTheme = createTheme({
  primaryColor: "dark",
  fontFamily: "monospace",
});

export const theme = mergeMantineTheme(DEFAULT_THEME, appTheme);
