import {
  alpha,
  lighten,
  parseThemeColor,
  useMantineTheme,
} from "@mantine/core";
import { IJoystickProps, Joystick } from "./Joystick";

export const ThemedJoystick = ({ ...othr }: IJoystickProps) => {
  const theme = useMantineTheme();
  const parsedColor = parseThemeColor({ color: theme.primaryColor, theme });

  return (
    <Joystick
      {...othr}
      baseColor={alpha(lighten(parsedColor.value, 0.4), 0.2)}
      thumbColor={alpha(lighten(parsedColor.value, 0.1), 0.8)}
    />
  );
};
