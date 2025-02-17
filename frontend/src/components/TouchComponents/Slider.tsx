import { createRef, useCallback, useEffect, useMemo, useState } from "react";
import { filterTouches, Vector2 } from "./utils";
import { useMantineTheme } from "@mantine/core";

export interface ISliderProps {
  baseWidth?: number;
  onChange?: (pos: number) => void;
  onStop?: () => void;
}

export const Slider = ({ baseWidth = 300, onChange, onStop }: ISliderProps) => {
  const baseRef = createRef<HTMLDivElement>();
  const thumbRef = createRef<HTMLDivElement>();
  const [touch, setTouch] = useState<Touch>();
  const [sliderPercent, setSliderPercent] = useState<number>(50.0);
  const [currentDelta, setCurrentDelta] = useState<number>(0.0);
  const [touchStartPos, setTouchStartPos] = useState<Vector2>({ x: 0, y: 0 });
  const theme = useMantineTheme();

  const computeClampedPercentage = useMemo(() => {
    return Math.max(0, Math.min(100, sliderPercent + currentDelta));
  }, [currentDelta, sliderPercent]);

  const touchStart = useCallback(
    (event: TouchEvent) => {
      // dont start tracking another finger if we are already tracking one
      event.preventDefault();
      if (touch) {
        return;
      }
      // if it is a new touch, start tracking it
      setTouchStartPos({
        x: event.touches[0].clientX,
        y: event.touches[0].clientY,
      });
      setTouch(event.touches[0]);
    },
    [touch],
  );

  const touchEnd = useCallback(() => {
    if (touch == undefined) {
      return;
    }
    setTouch(undefined);
    setSliderPercent(computeClampedPercentage);
    setCurrentDelta(0);
    // if (onChange) {
    //   onChange(getJoyPosition({ x: 0, y: 0 }));
    // }
    if (onStop) {
      onStop();
    }
  }, [computeClampedPercentage, onStop, touch]);

  const touchMove = useCallback(
    (event: TouchEvent) => {
      if (!touch || !baseRef.current) {
        return;
      }
      const touchFound = filterTouches(touch, event.changedTouches);
      if (touchFound) {
        const delta = {
          x: (touchFound.clientX - touchStartPos.x) / 5.0,
          y: (touchFound.clientY - touchStartPos.y) / 5.0,
        };
        // set a slight S curve to the delta
        const fact = Math.pow(delta.y / 25, 5);
        const currentDelta = delta.y / 1.5 + fact / 2;

        setCurrentDelta(currentDelta);
        if (onChange) {
          onChange(currentDelta);
        }
      }
    },
    [touch, baseRef, touchStartPos.x, touchStartPos.y, onChange],
  );

  useEffect(() => {
    const ref = baseRef.current;
    if (!ref) {
      return;
    }

    ref.addEventListener("touchstart", touchStart);
    ref.addEventListener("touchend", touchEnd);
    ref.addEventListener("touchmove", touchMove);
    ref.addEventListener("touchcancel", touchEnd);

    return () => {
      // remove event handlers
      ref.removeEventListener("touchstart", touchStart);
      ref.removeEventListener("touchend", touchEnd);
      ref.removeEventListener("touchmove", touchMove);
      ref.removeEventListener("touchcancel", touchEnd);
    };
  }, [baseRef, touchEnd, touchMove, touchStart]);

  return (
    <div
      ref={baseRef}
      style={{
        borderRadius: theme.radius["md"],
        backgroundColor: theme.colors[theme.primaryColor][5],
        width: baseWidth,
        height: "400px",
      }}
    >
      <div
        ref={thumbRef}
        style={{
          backgroundColor: theme.colors[theme.primaryColor][1],
          marginTop: "auto",
          width: "100%",
          height: `${computeClampedPercentage}%`,
        }}
      >
        {touch && touch.identifier}
      </div>
    </div>
  );
};
