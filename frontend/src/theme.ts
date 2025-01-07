import { createTheme, mergeMantineTheme, DEFAULT_THEME } from "@mantine/core";

export const appTheme = createTheme({
  primaryColor: "dark",
  fontFamily: "monospace",
  spacing: {
    xs: "4px",
    sm: "6px",
  },
});

export const theme = mergeMantineTheme(DEFAULT_THEME, appTheme);
