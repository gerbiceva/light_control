export interface Vector2 {
  x: number;
  y: number;
}
export const filterTouches = (touch: Touch, list: TouchList) => {
  for (let i = 0; i < list.length; i++) {
    const currTouch = list.item(i);
    if (currTouch?.identifier == touch?.identifier) {
      return currTouch;
    }
  }
};

export const limitVecSquare = (vec: Vector2, boundRect?: DOMRect): Vector2 => {
  if (!boundRect) {
    return { x: 0, y: 0 };
  }

  return {
    x: Math.min(Math.max(vec.x, boundRect.left), boundRect.right),
    y: Math.min(Math.max(vec.y, boundRect.top), boundRect.bottom),
  };
};

export const limitVecCircle = (vec: Vector2, boundRect?: DOMRect): Vector2 => {
  if (!boundRect) {
    return { x: 0, y: 0 };
  }

  const centerX = boundRect.left + boundRect.width / 2;
  const centerY = boundRect.top + boundRect.height / 2;
  const radius = Math.min(boundRect.width, boundRect.height) / 2;

  const dx = vec.x - centerX;
  const dy = vec.y - centerY;
  const distance = Math.sqrt(dx * dx + dy * dy);

  if (distance > radius) {
    const scale = radius / distance;
    return {
      x: centerX + dx * scale,
      y: centerY + dy * scale,
    };
  }

  return vec;
};

export const getJoyPosition = (vec: Vector2, boundRect?: DOMRect): Vector2 => {
  if (!boundRect) {
    return { x: 0, y: 0 };
  }

  const x =
    (((boundRect.left + boundRect.right) / 2.0 - vec.x) /
      (boundRect.width / 2.0)) *
    -1;
  const y =
    ((boundRect.top + boundRect.bottom) / 2.0 - vec.y) /
    (boundRect.height / 2.0);

  return {
    x,
    y,
  };
};

export const touchFix = () => {
  document.addEventListener(
    "dblclick",
    (event) => {
      event.preventDefault();
    },
    { passive: false },
  );

  // Prevent pinch-zoom
  document.addEventListener(
    "gesturestart",
    (event) => {
      event.preventDefault();
    },
    { passive: false },
  );

  // Prevent touch scrolling
  document.addEventListener(
    "touchmove",
    (event) => {
      event.preventDefault();
    },
    { passive: false },
  );

  document.addEventListener("contextmenu", function () {
    return false;
  });
};
