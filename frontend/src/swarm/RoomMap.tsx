import { useEffect, useRef } from "react";
import { drawMap, type Shown } from "./mapDraw";
import type { Finding, ZwarmState } from "./types";

interface Props {
  state: ZwarmState | null;
  findings: Finding[];
  staleS: number | null;
}

const EASE = 0.28;
const SNAP_M = 1.5;

const shortestTurn = (from: number, to: number) => {
  const turn = (to - from) % (Math.PI * 2);
  if (turn > Math.PI) return turn - Math.PI * 2;
  return turn < -Math.PI ? turn + Math.PI * 2 : turn;
};

// State arrives at 5 Hz. Each frame eases the drawn drones toward their newest pose, and
// the loop draws only when something has moved.
export default function RoomMap({ state, findings, staleS }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef({ state, findings, staleS });
  const dirty = useRef(true);

  useEffect(() => {
    latest.current = { state, findings, staleS };
    dirty.current = true;
  }, [state, findings, staleS]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const shown = new Map<string, Shown>();
    let width = 0;
    let height = 0;
    let dpr = 1;

    const resize = () => {
      const box = canvas.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1;
      width = box.width;
      height = box.height;
      canvas.width = Math.round(box.width * dpr);
      canvas.height = Math.round(box.height * dpr);
      dirty.current = true;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const ease = (current: ZwarmState) => {
      let moving = false;
      for (const d of current.drones) {
        if (!d.p) {
          shown.delete(d.id);
          continue;
        }
        const yawRad = (d.yawDeg * Math.PI) / 180;
        const pose = shown.get(d.id);
        if (!pose || Math.hypot(d.p[0] - pose.x, d.p[1] - pose.y) > SNAP_M) {
          shown.set(d.id, { x: d.p[0], y: d.p[1], yawRad });
          moving = true;
          continue;
        }
        const dx = d.p[0] - pose.x;
        const dy = d.p[1] - pose.y;
        const turn = shortestTurn(pose.yawRad, yawRad);
        if (Math.hypot(dx, dy) > 0.002 || Math.abs(turn) > 0.003) moving = true;
        pose.x += dx * EASE;
        pose.y += dy * EASE;
        pose.yawRad += turn * EASE;
      }
      return moving;
    };

    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const { state: current, findings: seen, staleS: stale } = latest.current;
      if (!current) return;
      const moving = ease(current);
      if (!moving && !dirty.current) return;
      dirty.current = false;
      try {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawMap(ctx, { width, height, state: current, findings: seen, shown, staleS: stale });
      } catch (error) {
        // A bad frame must not stop the next one.
        console.error("[zwarm] map draw failed", error);
      }
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="zw-map" role="img" aria-label="Top-down map of the room, the drones and what they have covered" />;
}
