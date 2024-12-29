import { appTheme } from "./theme.ts";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// core styles are required for all packages
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/spotlight/styles.css";

import { MantineProvider } from "@mantine/core";

import { MainLayout } from "./views/MainLayout.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider theme={appTheme} forceColorScheme="light">
      <MainLayout />
    </MantineProvider>
  </StrictMode>
);
