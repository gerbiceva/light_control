import { useEffect, useState } from "react";
import { $lastSync } from "../globalStore/loadingStore";
import { useStore } from "@nanostores/react";
import { timeElapsedPreety } from "./timeUtils";

export const useLastSync = () => {
  const lastSync = useStore($lastSync);
  const [preetyPrint, setPreetyPrint] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setPreetyPrint(timeElapsedPreety(lastSync));
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [lastSync]);

  return preetyPrint;
};
