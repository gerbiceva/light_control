import { MantineColorsTuple } from "@mantine/core";
import { theme } from "../theme";

export const getColorFromEnum = (num: number): MantineColorsTuple => {
  // we can assert that colors exist
  const cols = theme.colors;

  const keys = Object.keys(cols).slice(2);
  const color = cols[keys[(num * 13) % (keys.length - 1)]];
  return color!;
};
