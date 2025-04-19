// Enhanced GlobalSync.ts
import { atom } from "nanostores";
import { AppState } from "../globalStore/flowStore";
import { getYState2, initializeYState } from "./repo";
import { useStore } from "@nanostores/react";
import { useEffect, useState } from "react";

export const $syncedAppState = atom<AppState | undefined>(undefined);
export const $isStateReady = atom<boolean>(false);

export const initializeSyncedState = async () => {
  try {
    const initialized = await initializeYState();
    if (!initialized) throw new Error("Yjs initialization failed");

    const yState = getYState2();
    if (!yState) throw new Error("Yjs state not available");

    $syncedAppState.set(yState.toJSON() as AppState);
    $isStateReady.set(true);
  } catch (error) {
    console.error("State synchronization error:", error);
    $isStateReady.set(false);
  }
};

export const YSyncStore = {
  observe: () => {
    const yState = getYState2();
    if (!yState) {
      console.error("Yjs state not initialized");
      return () => {};
    }

    const observer = () => {
      try {
        const newState = yState.toJSON() as AppState;
        $syncedAppState.set(newState);
      } catch (error) {
        console.error("Sync error:", error);
      }
    };

    observer(); // Initial sync
    yState.observeDeep(observer);

    return () => yState.unobserveDeep(observer);
  },
};

// Hook to wait for state readiness
export function useYjsState() {
  const [isLoading, setIsLoading] = useState(true);
  const isReady = useStore($isStateReady);
  const appState = useStore($syncedAppState);

  useEffect(() => {
    if (!isReady) {
      initializeSyncedState().finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }

    return YSyncStore.observe();
  }, [isReady]);

  return {
    isLoading,
    isReady,
    appState,
    nodes: appState?.main?.nodes || [],
    edges: appState?.main?.edges || [],
    subgraphs: appState?.subgraphs || {},
  };
}
