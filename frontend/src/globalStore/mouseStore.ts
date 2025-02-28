import { atom } from "nanostores";

interface mousPos {
  x: number;
  y: number;
}
export const $mousePos = atom<mousPos>({ x: 0, y: 0 });

export const $frozenMousePos = atom<mousPos>({ x: 0, y: 0 });
export const freezeMousePos = () => {
  $frozenMousePos.set($mousePos.get());
};
