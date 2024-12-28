import { MantineColorsTuple } from "@mantine/core";
import { theme } from "../../../main";

export const getColorFromEnum = (num: number): MantineColorsTuple => {
  // we can assert that colors exist
  const cols = theme.colors;

  const keys = Object.keys(cols);
  const color = cols[keys[(num * 17) % (keys.length - 1)]];
  return color!;
};
