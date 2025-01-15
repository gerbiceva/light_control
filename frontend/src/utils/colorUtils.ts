import { MantineColorsTuple } from "@mantine/core";
import { theme } from "../theme";

export const getColorFromEnum = (num: number): MantineColorsTuple => {
  // we can assert that colors exist
  const cols = theme.colors;

  const keys = Object.keys(cols).slice(2);
  const color = cols[keys[(num * 13) % (keys.length - 1)]];
  return color!;
};

const hashString = (s: string): number => {
  let hash = 17;
  for (let i = 0; i < s.length; i++) {
    hash = (hash << 5) - hash + s.charCodeAt(i);
    hash |= 0; // Convert to 32-bit integer
  }
  return Math.abs(hash);
};

export const getColorFromString = (str: string): MantineColorsTuple => {
  // Simple string hashing function

  const hash = hashString(str);

  // Get Mantine colors from theme
  const cols = theme.colors;

  // Exclude specific colors (e.g., 'white' and 'black')
  const keys = Object.keys(cols).slice(2);

  // Select a color based on the hash value
  const colorKey = keys[hash % keys.length];
  const color = cols[colorKey];

  // Return the selected color tuple
  return color!;
};
