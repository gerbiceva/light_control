import { atom } from "nanostores";

export const $projectName = atom<string>("DefaultName");
export const setProjectName = (name: string) => {
  $projectName.set(name);
};
