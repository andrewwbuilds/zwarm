import { useEffect, useRef } from "react";
import type { Cloud } from "./types";

const AUTO_TURN_RAD_S = 0.25;
const DRAG_RAD_PER_PX = 0.008;

interface Fitted {
  count: number;
  xyz: Float32Array;
  rgb: Uint8ClampedArray;
}

// Centre on the median and scale by the 2nd to 98th percentile, so stray points far from the
// scan don't shrink the rest to a dot. COLMAP's own scale and origin are arbitrary.
function fit(cloud: Cloud): Fitted {
  const { count, xyz, rgb } = cloud;
  const center = [0, 0, 0];
  let span = 0;
  const axis = new Float32Array(count);
  for (let a = 0; a < 3; a += 1) {
    for (let i = 0; i < count; i += 1) axis[i] = xyz[i * 3 + a];
    axis.sort();
    center[a] = axis[Math.floor(count / 2)] ?? 0;
    span = Math.max(span, (axis[Math.floor(count * 0.98)] ?? 0) - (axis[Math.floor(count * 0.02)] ?? 0));
  }
  const scale = span > 0 ? 2 / span : 1;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    for (let a = 0; a < 3; a += 1) out[i * 3 + a] = (xyz[i * 3 + a] - center[a]) * scale;
  }
  return { count, xyz: out, rgb };
}

// Writes pixels straight into an ImageData with a depth buffer. Fast enough for 50,000 points
// on a plain 2D canvas, and no 3D library needed.
function render(ctx: CanvasRenderingContext2D, image: ImageData, depth: Float32Array, cloud: Fitted, yaw: number, pitch: number, zoom: number, splat: number): void {
  const { width, height, data } = image;
  data.fill(0);
  depth.fill(Infinity);
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);
  const focal = Math.min(width, height) * 0.95 * zoom;
  const cx = width / 2;
  const cy = height / 2;
  const { count, xyz, rgb } = cloud;
  for (let i = 0; i < count; i += 1) {
    const px = xyz[i * 3];
    const py = xyz[i * 3 + 1];
    const pz = xyz[i * 3 + 2];
    const rx = px * cosY - py * sinY;
    const ry = px * sinY + py * cosY;
    const up = pz * cosP - ry * sinP;
    const away = pz * sinP + ry * cosP + 3;
    if (away < 0.2) continue;
    const u = Math.round(cx + (focal * rx) / away);
    const v = Math.round(cy - (focal * up) / away);
    for (let ox = 0; ox < splat; ox += 1) {
      for (let oy = 0; oy < splat; oy += 1) {
        const x = u + ox;
        const y = v + oy;
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const at = y * width + x;
        if (away >= depth[at]) continue;
        depth[at] = away;
        const out = at * 4;
        data[out] = rgb[i * 3];
        data[out + 1] = rgb[i * 3 + 1];
        data[out + 2] = rgb[i * 3 + 2];
        data[out + 3] = 255;
      }
    }
  }
  ctx.putImageData(image, 0, 0);
}

export default function PointCloud({ cloud }: { cloud: Cloud }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const fitted = fit(cloud);
    let image: ImageData | null = null;
    let depth = new Float32Array(0);
    let splat = 2;
    let yaw = 0.7;
    let pitch = 0.45;
    let zoom = 1;
    let dirty = true;
    let dragging: { x: number; y: number } | null = null;
    let spinning = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(box.width * dpr));
      const h = Math.max(1, Math.round(box.height * dpr));
      canvas.width = w;
      canvas.height = h;
      image = ctx.createImageData(w, h);
      depth = new Float32Array(w * h);
      splat = dpr >= 1.5 ? 3 : 2;
      dirty = true;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const onDown = (e: PointerEvent) => {
      dragging = { x: e.clientX, y: e.clientY };
      spinning = false;
      canvas.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      yaw += (e.clientX - dragging.x) * DRAG_RAD_PER_PX;
      pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - dragging.y) * DRAG_RAD_PER_PX));
      dragging = { x: e.clientX, y: e.clientY };
      dirty = true;
    };
    const onUp = () => {
      dragging = null;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoom = Math.max(0.4, Math.min(6, zoom * Math.exp(-e.deltaY * 0.0015)));
      dirty = true;
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    let frame = 0;
    let last = performance.now();
    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (spinning) {
        yaw += AUTO_TURN_RAD_S * dt;
        dirty = true;
      }
      if (!dirty || !image) return;
      dirty = false;
      const started = performance.now();
      render(ctx, image, depth, fitted, yaw, pitch, zoom, splat);
      canvas.dataset.renderMs = (performance.now() - started).toFixed(1);
    };
    frame = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [cloud]);

  return <canvas ref={canvasRef} className="zw-cloud" role="img" aria-label="3D point cloud of the room. Drag to orbit, scroll to zoom." />;
}
