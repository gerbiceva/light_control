import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// core styles are required for all packages
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/spotlight/styles.css";

import { createTheme, MantineProvider } from "@mantine/core";
import { MainLayout } from "./views/MainLayout.tsx";

const theme = createTheme({
  primaryColor: "dark",
  fontFamily: "monospace",
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider theme={theme} forceColorScheme="light">
      <MainLayout />
    </MantineProvider>
  </StrictMode>
);
