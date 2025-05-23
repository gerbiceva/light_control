import { createRef, useCallback, useEffect, useRef } from "react";
import { filterTouches, Vector2 } from "./utils";
import { alpha, useMantineTheme } from "@mantine/core";

export interface ISliderProps {
  baseWidth?: number | string;
  onChange?: (pos: number) => void;
  onStop?: () => void;
}

export const Slider = ({ baseWidth = 300, onChange, onStop }: ISliderProps) => {
  const baseRef = createRef<HTMLDivElement>();
  const thumbRef = createRef<HTMLDivElement>();
  const touch = useRef<Touch | null>(null);
  const touchStartPos = useRef<Vector2>({ x: 0, y: 0 });
  const sliderPercent = useRef(50);
  const currentDelta = useRef(0);
  const theme = useMantineTheme();
  const animationFrameId = useRef<number | null>(null);

  const updateThumbPosition = useCallback(() => {
    if (!thumbRef.current) return;

    const clampedPercentage = Math.max(
      0,
      Math.min(100, sliderPercent.current + currentDelta.current)
    );

    thumbRef.current.style.height = `${clampedPercentage}%`;
  }, [thumbRef]);

  // Touch handlers
  const touchStart = useCallback((event: TouchEvent) => {
    event.preventDefault();
    if (touch.current) return;

    touchStartPos.current = {
      x: event.touches[0].clientX,
      y: event.touches[0].clientY,
    };
    touch.current = event.touches[0];
  }, []);

  const touchEnd = useCallback(() => {
    if (!touch.current) return;

    touch.current = null;
    const newPercent = Math.max(
      0,
      Math.min(100, sliderPercent.current + currentDelta.current)
    );
    sliderPercent.current = newPercent;
    currentDelta.current = 0;
    updateThumbPosition();
    onStop?.();
  }, [onStop, updateThumbPosition]);

  const touchMove = useCallback(
    (event: TouchEvent) => {
      if (!touch.current || !baseRef.current) return;
      const touchFound = filterTouches(touch.current, event.changedTouches);
      if (touchFound) {
        const delta = {
          x: (touchFound.clientX - touchStartPos.current.x) / 5,
          y: (touchFound.clientY - touchStartPos.current.y) / 5,
        };

        const fact = Math.pow(delta.y / 8, 3);
        currentDelta.current = (delta.y / 1.8 + fact / 3) * -1;
        const newPercentage = Math.max(
          0,
          Math.min(100, sliderPercent.current + currentDelta.current)
        );

        if (animationFrameId.current) {
          cancelAnimationFrame(animationFrameId.current);
        }

        animationFrameId.current = requestAnimationFrame(() => {
          onChange?.(newPercentage);
          updateThumbPosition();
        });
      }
    },
    [baseRef, onChange, updateThumbPosition]
  );

  // Mouse handlers
  const handleMouseDown = useCallback((event: MouseEvent) => {
    event.preventDefault();
    if (touch.current) return;

    touchStartPos.current = {
      x: event.clientX,
      y: event.clientY,
    };
    touch.current = { identifier: -1 } as Touch;
  }, []);

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!touch.current || !baseRef.current) return;

      const delta = {
        x: (event.clientX - touchStartPos.current.x) / 2,
        y: (event.clientY - touchStartPos.current.y) / 2,
      };

      currentDelta.current = delta.y * -1;
      const newPercentage = Math.max(
        0,
        Math.min(100, sliderPercent.current + currentDelta.current)
      );

      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }

      animationFrameId.current = requestAnimationFrame(() => {
        onChange?.(newPercentage);
        updateThumbPosition();
      });
    },
    [baseRef, onChange, updateThumbPosition]
  );

  const handleMouseUp = useCallback(() => {
    if (!touch.current) return;

    touch.current = null;
    const newPercent = Math.max(
      0,
      Math.min(100, sliderPercent.current + currentDelta.current)
    );
    sliderPercent.current = newPercent;
    currentDelta.current = 0;
    updateThumbPosition();
    onStop?.();
  }, [onStop, updateThumbPosition]);

  useEffect(() => {
    const ref = baseRef.current;
    if (!ref) return;

    const options = { passive: false };
    // Touch event listeners
    ref.addEventListener("touchstart", touchStart, options);
    ref.addEventListener("touchend", touchEnd, options);
    ref.addEventListener("touchmove", touchMove, options);
    ref.addEventListener("touchcancel", touchEnd, options);

    // Mouse event listeners
    ref.addEventListener("mousedown", handleMouseDown, options);
    ref.addEventListener("mousemove", handleMouseMove, options);
    ref.addEventListener("mouseup", handleMouseUp, options);
    ref.addEventListener("mouseleave", handleMouseUp, options);

    return () => {
      // Cleanup touch listeners
      ref.removeEventListener("touchstart", touchStart);
      ref.removeEventListener("touchend", touchEnd);
      ref.removeEventListener("touchmove", touchMove);
      ref.removeEventListener("touchcancel", touchEnd);

      // Cleanup mouse listeners
      ref.removeEventListener("mousedown", handleMouseDown);
      ref.removeEventListener("mousemove", handleMouseMove);
      ref.removeEventListener("mouseup", handleMouseUp);
      ref.removeEventListener("mouseleave", handleMouseUp);

      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [
    baseRef,
    touchStart,
    touchEnd,
    touchMove,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
  ]);

  return (
    <div
      ref={baseRef}
      style={{
        borderRadius: theme.radius.md,
        backgroundColor: alpha(theme.colors["cyan"][4], 0.2),
        width: baseWidth,
        height: "100%",
        position: "relative",
        display: "flex",
        justifyContent: "flex-start",
        alignItems: "center",
        flexDirection: "column",
      }}
    >
      <div
        ref={thumbRef}
        style={{
          backgroundColor: theme.colors["cyan"][7],
          marginTop: "auto",
          width: "100%",
          height: `${sliderPercent.current}%`,
        }}
      />
    </div>
  );
};
