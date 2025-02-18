import { createRef, useCallback, useEffect, useState } from "react";
import {
  filterTouches,
  getJoyPosition,
  limitVecCircle,
  Vector2,
} from "./utils";

export interface IJoystickProps {
  baseSize?: number;
  baseColor?: string;
  thumbSize?: number;
  thumbColor?: string;
  onChange?: (pos: Vector2) => void;
  onStop?: () => void;
}

export const Joystick = ({
  baseSize = 300,
  baseColor = "yellow",
  thumbSize = 50,
  thumbColor = "red",
  onChange,
  onStop,
}: IJoystickProps) => {
  const baseRef = createRef<HTMLDivElement>();
  const thumbRef = createRef<HTMLDivElement>();
  const [touch, setTouch] = useState<Touch>();
  const [joyPos, setJoyPos] = useState<Vector2>({ x: 0, y: 0 });
  const [joyCenter, setJoyCenter] = useState<Vector2>({ x: 0, y: 0 });

  const touchStart = useCallback(
    (event: TouchEvent) => {
      // dont start tracking another finger if we are already tracking one
      event.preventDefault();
      if (touch) {
        return;
      }
      // if it is a new touch, start tracking it
      setTouch(event.changedTouches[0]);
      setJoyPos({
        x: event.changedTouches[0].clientX,
        y: event.changedTouches[0].clientY,
      });
    },
    [touch]
  );

  const touchEnd = useCallback(() => {
    if (touch == undefined) {
      return;
    }
    setTouch(undefined);
    setJoyPos(joyCenter);
    // if (onChange) {
    //   onChange(getJoyPosition({ x: 0, y: 0 }));
    // }
    if (onStop) {
      onStop();
    }
  }, [joyCenter, onStop, touch]);

  const touchMove = useCallback(
    (event: TouchEvent) => {
      if (!touch || !baseRef.current) {
        return;
      }
      const touchFound = filterTouches(touch, event.changedTouches);
      if (touchFound) {
        const limited = limitVecCircle(
          {
            x: touchFound.clientX,
            y: touchFound.clientY,
          },
          baseRef.current.getBoundingClientRect()
        );
        setJoyPos(limited);
        if (onChange) {
          onChange(
            getJoyPosition(limited, baseRef.current.getBoundingClientRect())
          );
        }
      }
    },
    [touch, baseRef, onChange]
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

  useEffect(() => {
    const ref = baseRef.current;
    if (!ref) {
      return;
    }

    const r = baseRef.current.getBoundingClientRect();
    const center = {
      x: (r.left + r.right) / 2.0,
      y: (r.top + r.bottom) / 2.0,
    };
    setJoyPos(center);
    setJoyCenter(center);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseRef.current, baseSize]);

  return (
    <div
      ref={baseRef}
      style={{
        borderRadius: "100%",
        backgroundColor: baseColor,
        width: baseSize,
        height: baseSize,
      }}
    >
      {touch && touch.identifier}
      <div
        ref={thumbRef}
        style={{
          position: "absolute",
          left: joyPos.x - thumbSize / 2,
          top: joyPos.y - thumbSize / 2,
          borderRadius: "100%",
          backgroundColor: thumbColor,
          width: thumbSize,
          height: thumbSize,
        }}
      />
    </div>
  );
};
