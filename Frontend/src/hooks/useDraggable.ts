import { useRef, useState } from "react";
import type React from "react";

/**
 * Lets a floating, fixed-position banner be dragged out of the way of the
 * controls it covers. Spread `dragProps` on the banner's root element and
 * append `dragClassName` to its classes.
 *
 * The banner keeps its docked CSS position; the drag is applied on top with
 * the `translate` property, which leaves the enter animation's `transform`
 * alone. Position is per mount, so a reload docks it again.
 */
export const useDraggable = <T extends HTMLElement = HTMLDivElement>() => {
  // Distance moved from the docked position.
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const ref = useRef<T>(null);
  const dragStart = useRef<{ pointerX: number; pointerY: number; x: number; y: number; rect: DOMRect } | null>(null);

  const onPointerDown = (e: React.PointerEvent<T>) => {
    // Leave the banner's own buttons clickable.
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    dragStart.current = { pointerX: e.clientX, pointerY: e.clientY, x: offset.x, y: offset.y, rect };
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<T>) => {
    const start = dragStart.current;
    if (!start) return;
    // Clamp so the whole banner stays inside the viewport and can be grabbed again.
    const dx = Math.min(
      Math.max(e.clientX - start.pointerX, -start.rect.left),
      window.innerWidth - start.rect.right,
    );
    const dy = Math.min(
      Math.max(e.clientY - start.pointerY, -start.rect.top),
      window.innerHeight - start.rect.bottom,
    );
    setOffset({ x: start.x + dx, y: start.y + dy });
  };

  const endDrag = () => {
    dragStart.current = null;
    setDragging(false);
  };

  return {
    dragProps: {
      ref,
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      title: "Drag to move",
      style: { translate: `${offset.x}px ${offset.y}px`, touchAction: "none" } as React.CSSProperties,
    },
    dragClassName: `select-none ${dragging ? "cursor-grabbing" : "cursor-grab"}`,
  };
};
