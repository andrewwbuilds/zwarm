// Draws the top-down room plan: north (+x) up, west (+y) left, like a floor plan on a wall.

import { droneColor, isFlying, trackingQuality } from "./format";
import type { Drone, Finding, Room, ZwarmState } from "./types";

export interface Shown {
  x: number;
  y: number;
  yawRad: number;
}

export interface MapScene {
  width: number;
  height: number;
  state: ZwarmState;
  findings: Finding[];
  shown: Map<string, Shown>;
  staleS: number | null;
}

const CHALK = "#eef5f7";
const SOFT = "#b4c9d8";
const GROUND = "#0f2f52";
const YELLOW = "#ffd100";
const FAULT = "#ff9c8f";
const LABEL_FONT = '600 13px "Avenir Next", "Segoe UI", system-ui, sans-serif';
const COMPASS_FONT = '700 20px "DIN Condensed", "DIN Alternate", Bahnschrift, "Arial Narrow", sans-serif';
const MARGIN = 34;
const FOOTPRINT_FAR_M = 2.4;
const TILT_RAD = Math.PI / 4;
const HALF_VFOV_RAD = (29 * Math.PI) / 180;

interface Frame {
  s: number;
  sx: (y: number) => number;
  sy: (x: number) => number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function makeFrame(room: Room, width: number, height: number): Frame {
  const spanX = room.hi[0] - room.lo[0];
  const spanY = room.hi[1] - room.lo[1];
  const s = Math.max(1, Math.min((width - 2 * MARGIN) / spanY, (height - 2 * MARGIN) / spanX));
  const left = (width - spanY * s) / 2;
  const top = (height - spanX * s) / 2;
  return {
    s,
    sx: (y) => left + (room.hi[1] - y) * s,
    sy: (x) => top + (room.hi[0] - x) * s,
    left,
    right: left + spanY * s,
    top,
    bottom: top + spanX * s,
  };
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string): void {
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.strokeStyle = GROUND;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function hatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = "rgba(238,245,247,0.42)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let d = -h; d < w; d += 8) {
    ctx.moveTo(x + d, y + h);
    ctx.lineTo(x + d + h, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawGrid(ctx: CanvasRenderingContext2D, room: Room, f: Frame): void {
  ctx.fillStyle = "#123a63";
  ctx.fillRect(f.left, f.top, f.right - f.left, f.bottom - f.top);
  ctx.strokeStyle = "rgba(238,245,247,0.09)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = Math.ceil(room.lo[0]); x <= room.hi[0]; x += 1) {
    const py = Math.round(f.sy(x)) + 0.5;
    ctx.moveTo(f.left, py);
    ctx.lineTo(f.right, py);
  }
  for (let y = Math.ceil(room.lo[1]); y <= room.hi[1]; y += 1) {
    const px = Math.round(f.sx(y)) + 0.5;
    ctx.moveTo(px, f.top);
    ctx.lineTo(px, f.bottom);
  }
  ctx.stroke();
}

function drawCoverage(ctx: CanvasRenderingContext2D, state: ZwarmState, f: Frame): void {
  const coverage = state.coverage;
  if (!coverage) return;
  const half = coverage.cell / 2;
  for (const [x, y, seen] of coverage.cells) {
    if (seen <= 0) continue;
    const x0 = Math.round(f.sx(y + half));
    const x1 = Math.round(f.sx(y - half));
    const y0 = Math.round(f.sy(x + half));
    const y1 = Math.round(f.sy(x - half));
    ctx.fillStyle = `rgba(150,214,255,${(0.1 + 0.34 * Math.min(1, seen)).toFixed(3)})`;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }
}

// Half-plane rectangles in a frame where -y points north and +x points east, centred on the room.
function regionRect(region: string, room: Room, s: number): [number, number, number, number] | null {
  const R = 5000;
  switch (region) {
    case "north": return [-R, -R, 2 * R, R];
    case "south": return [-R, 0, 2 * R, R];
    case "east": return [0, -R, R, 2 * R];
    case "west": return [-R, -R, R, 2 * R];
    case "north-east": return [0, -R, R, R];
    case "north-west": return [-R, -R, R, R];
    case "south-east": return [0, 0, R, R];
    case "south-west": return [-R, 0, R, R];
    case "center": {
      const w = ((room.hi[1] - room.lo[1]) * s) / 2;
      const h = ((room.hi[0] - room.lo[0]) * s) / 2;
      return [-w / 2, -h / 2, w, h];
    }
    default: return null;
  }
}

function drawRegion(ctx: CanvasRenderingContext2D, state: ZwarmState, room: Room, f: Frame): void {
  const mission = state.mission;
  if (!mission?.kind) return;
  const rect = regionRect(mission.region.replace(/_/g, "-"), room, f.s);
  if (!rect) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(f.left, f.top, f.right - f.left, f.bottom - f.top);
  ctx.clip();
  ctx.translate((f.left + f.right) / 2, (f.top + f.bottom) / 2);
  ctx.rotate((-room.northYawDeg * Math.PI) / 180);
  ctx.fillStyle = "rgba(255,209,0,0.09)";
  ctx.fillRect(...rect);
  ctx.strokeStyle = YELLOW;
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 7]);
  ctx.strokeRect(...rect);
  ctx.restore();
}

function drawObstacles(ctx: CanvasRenderingContext2D, room: Room, f: Frame): void {
  ctx.font = LABEL_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const o of room.obstacles) {
    const x = f.sx(o.hi[1]);
    const y = f.sy(o.hi[0]);
    const w = (o.hi[1] - o.lo[1]) * f.s;
    const h = (o.hi[0] - o.lo[0]) * f.s;
    ctx.fillStyle = GROUND;
    ctx.fillRect(x, y, w, h);
    hatch(ctx, x, y, w, h);
    ctx.strokeStyle = CHALK;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);
    // Names like "furniture-3" are generated from the scene and only add noise.
    if (!/-\d+$/.test(o.name)) label(ctx, o.name, x + w / 2, y + h / 2, CHALK);
  }
}

function drawAnchors(ctx: CanvasRenderingContext2D, room: Room, f: Frame): void {
  const size = Math.max(8, 0.15 * f.s);
  ctx.font = '600 11px "Avenir Next", "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const a of room.anchors) {
    const x = f.sx(a.y);
    const y = f.sy(a.x);
    ctx.fillStyle = CHALK;
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
    ctx.fillStyle = GROUND;
    ctx.fillRect(x - size / 4, y - size / 4, size / 4, size / 4);
    ctx.fillRect(x + size / 16, y + size / 16, size / 4, size / 4);
    label(ctx, a.id.replace(/^(aruco|qr):/, ""), x + size / 2 + 3, y, SOFT);
  }
}

function drawThings(ctx: CanvasRenderingContext2D, state: ZwarmState, f: Frame): void {
  ctx.font = LABEL_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const t of state.things) {
    const x = f.sx(t.y);
    const y = f.sy(t.x);
    ctx.beginPath();
    ctx.moveTo(x, y - 7);
    ctx.lineTo(x + 7, y);
    ctx.lineTo(x, y + 7);
    ctx.lineTo(x - 7, y);
    ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = CHALK;
    ctx.fillStyle = t.sure ? CHALK : GROUND;
    ctx.fill();
    ctx.stroke();
    label(ctx, t.label, x + 11, y, CHALK);
  }
}

function drawFindings(ctx: CanvasRenderingContext2D, findings: Finding[], f: Frame): void {
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const finding of findings) {
    if (finding.x === null || finding.y === null || finding.status === "dismissed") continue;
    const x = f.sx(finding.y);
    const y = f.sy(finding.x);
    ctx.beginPath();
    ctx.moveTo(x, y - 12);
    ctx.lineTo(x + 12, y + 9);
    ctx.lineTo(x - 12, y + 9);
    ctx.closePath();
    ctx.fillStyle = finding.status === "confirmed" ? "#e01b12" : YELLOW;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = finding.status === "confirmed" ? "#fff" : "#000";
    ctx.font = '800 14px "Avenir Next", system-ui, sans-serif';
    ctx.fillText("!", x, y + 2);
  }
}

// The patch of floor the camera sees: tilted 45 degrees down, about 90 by 58 degrees of view.
function drawFootprint(ctx: CanvasRenderingContext2D, d: Drone, pose: Shown, color: string, f: Frame): void {
  const z = d.p?.[2] ?? 0;
  if (z < 0.2 || !isFlying(d.phase)) return;
  const near = z / Math.tan(TILT_RAD + HALF_VFOV_RAD);
  const far = Math.min(z / Math.tan(TILT_RAD - HALF_VFOV_RAD), FOOTPRINT_FAR_M);
  const halfAt = (ahead: number) => (ahead + z) * Math.SQRT1_2;
  const cos = Math.cos(pose.yawRad);
  const sin = Math.sin(pose.yawRad);
  const corner = (ahead: number, left: number): [number, number] => [
    f.sx(pose.y + ahead * sin + left * cos),
    f.sy(pose.x + ahead * cos - left * sin),
  ];
  ctx.save();
  ctx.beginPath();
  ctx.rect(f.left, f.top, f.right - f.left, f.bottom - f.top);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(...corner(near, halfAt(near)));
  ctx.lineTo(...corner(far, halfAt(far)));
  ctx.lineTo(...corner(far, -halfAt(far)));
  ctx.lineTo(...corner(near, -halfAt(near)));
  ctx.closePath();
  ctx.globalAlpha = 0.14;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 0.45;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

function drawPath(ctx: CanvasRenderingContext2D, d: Drone, pose: Shown, color: string, f: Frame): void {
  const points = d.path.length ? d.path : d.goal ? [d.goal] : [];
  if (points.length === 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(f.sx(pose.y), f.sy(pose.x));
  for (const p of points) ctx.lineTo(f.sx(p[1]), f.sy(p[0]));
  ctx.stroke();
  ctx.setLineDash([]);
  const end = points[points.length - 1];
  const x = f.sx(end[1]);
  const y = f.sy(end[0]);
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawDrone(ctx: CanvasRenderingContext2D, d: Drone, pose: Shown, color: string, f: Frame): void {
  const x = f.sx(pose.y);
  const y = f.sy(pose.x);
  const quality = trackingQuality(d);

  if (d.sigma !== null) {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(6, d.sigma * f.s), 0, Math.PI * 2);
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = quality === "good" || quality === "fair" ? color : FAULT;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(quality === "good" ? [] : [4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  ctx.save();
  ctx.translate(x, y);
  // Screen angle of the nose. World yaw 0 points up the page and turns counter-clockwise.
  ctx.rotate(-pose.yawRad - Math.PI / 2);
  const flying = isFlying(d.phase);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (const [ax, ay] of [[9, 9], [9, -9], [-9, 9], [-9, -9]]) {
    ctx.moveTo(0, 0);
    ctx.lineTo(ax, ay);
  }
  ctx.stroke();
  for (const [ax, ay] of [[9, 9], [9, -9], [-9, 9], [-9, -9]]) {
    ctx.beginPath();
    ctx.arc(ax, ay, 5, 0, Math.PI * 2);
    ctx.fillStyle = flying ? color : GROUND;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(22, 0);
  ctx.lineTo(12, 6);
  ctx.lineTo(12, -6);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();

  ctx.font = '700 16px "Avenir Next", "Segoe UI", system-ui, sans-serif';
  ctx.textBaseline = "middle";
  const broken = d.faults.length > 0 || !d.connected;
  const roomOnRight = x + 18 + ctx.measureText(d.id).width < f.right - 4;
  ctx.textAlign = roomOnRight ? "left" : "right";
  label(ctx, d.id, roomOnRight ? x + 18 : x - 18, y - 16, broken ? FAULT : color);
}

function drawCompass(ctx: CanvasRenderingContext2D, room: Room, f: Frame): void {
  const yaw = (room.northYawDeg * Math.PI) / 180;
  const cx = (f.left + f.right) / 2;
  const cy = (f.top + f.bottom) / 2;
  const halfW = (f.right - f.left) / 2 + 16;
  const halfH = (f.bottom - f.top) / 2 + 16;
  ctx.font = COMPASS_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Screen direction of each compass point: north is yaw on the plan, the rest follow clockwise.
  for (const [letter, turn] of [["N", 0], ["E", 1], ["S", 2], ["W", 3]] as const) {
    const angle = -yaw + (turn * Math.PI) / 2;
    const dx = Math.sin(angle);
    const dy = -Math.cos(angle);
    const reach = Math.min(Math.abs(dx) > 1e-6 ? halfW / Math.abs(dx) : Infinity, Math.abs(dy) > 1e-6 ? halfH / Math.abs(dy) : Infinity);
    ctx.fillStyle = letter === "N" ? YELLOW : SOFT;
    ctx.fillText(letter, cx + dx * reach, cy + dy * reach);
  }
}

// In the bottom corner of the canvas, clear of the S on the compass.
function drawScale(ctx: CanvasRenderingContext2D, room: Room, f: Frame, height: number): void {
  const x = 14;
  const y = height - 16;
  ctx.strokeStyle = CHALK;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y - 4);
  ctx.lineTo(x, y + 4);
  ctx.moveTo(x, y);
  ctx.lineTo(x + f.s, y);
  ctx.moveTo(x + f.s, y - 4);
  ctx.lineTo(x + f.s, y + 4);
  ctx.stroke();
  ctx.font = LABEL_FONT;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillStyle = SOFT;
  const northSouth = room.hi[0] - room.lo[0];
  const eastWest = room.hi[1] - room.lo[1];
  ctx.fillText("1 m", x + f.s + 8, y);
  ctx.fillText(`Room ${eastWest.toFixed(1)} by ${northSouth.toFixed(1)} m`, x, y - 20);
}

function drawStale(ctx: CanvasRenderingContext2D, width: number, height: number, staleS: number): void {
  ctx.fillStyle = "rgba(10,35,64,0.62)";
  ctx.fillRect(0, 0, width, height);
  ctx.font = '700 28px "DIN Condensed", "DIN Alternate", Bahnschrift, "Arial Narrow", sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = YELLOW;
  ctx.fillText(`Last known positions, ${Math.round(staleS)} s old`, width / 2, height / 2);
}

export function drawMap(ctx: CanvasRenderingContext2D, scene: MapScene): void {
  const { width, height, state, findings, shown, staleS } = scene;
  ctx.clearRect(0, 0, width, height);
  const room = state.room;
  if (!room) {
    ctx.fillStyle = SOFT;
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.fillText("The backend hasn't sent a room yet.", width / 2, height / 2);
    return;
  }
  const f = makeFrame(room, width, height);
  drawGrid(ctx, room, f);
  drawCoverage(ctx, state, f);
  drawRegion(ctx, state, room, f);
  drawObstacles(ctx, room, f);
  drawAnchors(ctx, room, f);
  drawThings(ctx, state, f);
  drawFindings(ctx, findings, f);
  state.drones.forEach((d, i) => {
    const pose = shown.get(d.id);
    if (pose) drawFootprint(ctx, d, pose, droneColor(i), f);
  });
  state.drones.forEach((d, i) => {
    const pose = shown.get(d.id);
    if (pose) drawPath(ctx, d, pose, droneColor(i), f);
  });
  ctx.strokeStyle = CHALK;
  ctx.lineWidth = 3;
  ctx.strokeRect(f.left, f.top, f.right - f.left, f.bottom - f.top);
  state.drones.forEach((d, i) => {
    const pose = shown.get(d.id);
    if (pose) drawDrone(ctx, d, pose, droneColor(i), f);
  });
  drawCompass(ctx, room, f);
  drawScale(ctx, room, f, height);
  if (staleS !== null) drawStale(ctx, width, height, staleS);
}
