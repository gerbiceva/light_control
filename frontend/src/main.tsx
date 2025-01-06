import { appTheme } from "./theme.ts";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// core styles are required for all packages
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/spotlight/styles.css";
import "@mantine/notifications/styles.css";

import { MantineProvider } from "@mantine/core";

import { MainLayout } from "./views/MainLayout.tsx";
import { Notifications } from "@mantine/notifications";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider theme={appTheme} forceColorScheme="light">
      <Notifications position="bottom-center" />
      <MainLayout />
    </MantineProvider>
  </StrictMode>
);
