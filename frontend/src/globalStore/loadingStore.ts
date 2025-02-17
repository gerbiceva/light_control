import { atom, computed } from "nanostores";

interface syncTrack {
  lastSync: Date;
  lastChange: Date;
  autoUpdate: boolean;
}

export const $sync = atom<syncTrack>({
  lastChange: new Date(),
  lastSync: new Date(),
  autoUpdate: true,
});

export const setAutoUpdate = (autoUpdate: boolean) => {
  $sync.set({
    lastChange: $sync.get().lastChange,
    lastSync: $sync.get().lastSync,
    autoUpdate,
  });
};

export const $isSyncing = computed($sync, (sync) => {
  return sync.lastChange.getTime() >= sync.lastSync.getTime();
});

export const $lastSync = computed($sync, (sync) => {
  return sync.lastSync;
});

export const syncFinished = () => {
  $sync.set({
    lastChange: $sync.get().lastChange,
    lastSync: new Date(),
    autoUpdate: $sync.get().autoUpdate,
  });
};

export const changeHappened = () => {
  $sync.set({
    lastChange: new Date(),
    lastSync: $sync.get().lastSync,
    autoUpdate: $sync.get().autoUpdate,
  });
};

export const addSyncPromise = (promise: Promise<unknown>) => {
  changeHappened();
  // this needs BOTH then AND finally because grpc resolves this in a very wierd way
  promise.then(syncFinished).finally(syncFinished);
};
