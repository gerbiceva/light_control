import { atom, computed } from "nanostores";

interface syncTrack {
  lastSync: Date;
  lastChange: Date;
}

const $sync = atom<syncTrack>({
  lastChange: new Date(),
  lastSync: new Date(),
});

export const $isSyncing = computed($sync, (sync) => {
  return sync.lastChange.getTime() >= sync.lastSync.getTime();
});

export const syncFinished = () => {
  $sync.set({
    lastChange: $sync.get().lastChange,
    lastSync: new Date(),
  });
};

export const changeHappened = () => {
  $sync.set({
    lastChange: new Date(),
    lastSync: $sync.get().lastSync,
  });
};

export const addSyncPromise = (promise: Promise<unknown>) => {
  changeHappened();
  promise.finally(() => {
    syncFinished();
  });
};
