// Turns backend JSON into the types in types.ts. A missing or odd field becomes a safe
// default instead of a crash, because the backend is changing while this console is in use.
// Text from the backend is only ever shown as text.

import {
  EVENT_KINDS,
  type AgentInfo, type Anchor, type ChatReply, type Cloud, type Coverage, type CoverageCell, type Drone,
  type Decision, type EventKind, type Finding, type FindingStatus, type LogEvent, type MapStatus, type Mission, type Obstacle,
  type Plan, type Room, type Thing, type ToolCall, type Vec3, type ZwarmState,
} from "./types";

type Json = Record<string, unknown>;

export const DEFAULT_FLIGHT_LIMIT_S = 270;

const isRecord = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const record = (v: unknown): Json => (isRecord(v) ? v : {});
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const text = (v: unknown): string | null => {
  if (typeof v === "string") return v;
  return typeof v === "number" ? String(v) : null;
};

function firstText(source: Json, keys: readonly string[]): string | null {
  for (const key of keys) {
    const value = text(source[key]);
    if (value) return value;
  }
  return null;
}

function vec3(v: unknown): Vec3 | null {
  if (!Array.isArray(v)) return null;
  const x = num(v[0]);
  const y = num(v[1]);
  if (x === null || y === null) return null;
  return [x, y, num(v[2]) ?? 0];
}

// A short readable form of a tool's arguments or result.
function brief(v: unknown, limit = 160): string | null {
  if (v === undefined || v === null || v === "") return null;
  const out = typeof v === "string" ? v : JSON.stringify(v);
  return out.length > limit ? `${out.slice(0, limit - 1)}…` : out;
}

function parseDrone(v: unknown, flightById: Json): Drone | null {
  const d = record(v);
  const id = text(d.id);
  if (id === null) return null;
  return {
    id,
    phase: text(d.phase) ?? "unknown",
    p: vec3(d.p),
    yawDeg: num(d.yaw) ?? 0,
    sigma: num(d.sigma),
    fixAge: num(d.fix_age) ?? num(d.pose_age),
    flightS: num(d.flight_s) ?? num(flightById[id]) ?? 0,
    faults: list(d.faults).map((f) => text(f) ?? brief(f) ?? "fault"),
    connected: typeof d.connected === "boolean" ? d.connected : d.link !== false,
    path: list(d.path).map(vec3).filter((p) => p !== null),
    goal: vec3(d.goal),
  };
}

function parseMission(v: unknown): Mission | null {
  if (!isRecord(v)) return null;
  const tasks: Record<string, string> = {};
  for (const [drone, task] of Object.entries(record(v.tasks))) tasks[drone] = text(task) ?? brief(task) ?? "";
  const flightS: Record<string, number> = {};
  for (const [drone, seconds] of Object.entries(record(v.flight_s))) flightS[drone] = num(seconds) ?? 0;
  return {
    kind: text(v.kind),
    state: text(v.state) ?? "idle",
    region: text(v.region) ?? "all",
    shareSeen: num(v.share_seen) ?? 0,
    target: text(v.target),
    elapsedS: num(v.elapsed_s) ?? 0,
    tasks,
    flightS,
  };
}

function parseObstacle(v: unknown, index: number): Obstacle | null {
  const o = record(v);
  const lo = vec3(o.lo);
  const hi = vec3(o.hi);
  if (!lo || !hi) return null;
  return { name: text(o.name) ?? `obstacle-${index}`, lo, hi };
}

function parseAnchor(v: unknown, fallbackId: string): Anchor | null {
  if (Array.isArray(v)) {
    const p = vec3(v);
    return p && { id: fallbackId, x: p[0], y: p[1] };
  }
  const a = record(v);
  const p = vec3(a.p);
  const x = num(a.x) ?? p?.[0] ?? null;
  const y = num(a.y) ?? p?.[1] ?? null;
  if (x === null || y === null) return null;
  return { id: text(a.id) ?? fallbackId, x, y };
}

function parseAnchors(v: unknown): Anchor[] {
  const entries = Array.isArray(v)
    ? v.map((item, i): [string, unknown] => [String(i), item])
    : Object.entries(record(v));
  return entries.map(([key, item]) => parseAnchor(item, key)).filter((a) => a !== null);
}

function parseRoom(v: unknown): Room | null {
  if (!isRecord(v)) return null;
  const fence = record(v.fence);
  const lo = vec3(v.lo) ?? vec3(fence.lo);
  const hi = vec3(v.hi) ?? vec3(fence.hi);
  if (!lo || !hi || hi[0] <= lo[0] || hi[1] <= lo[1]) return null;
  return {
    lo,
    hi,
    obstacles: list(v.obstacles).map(parseObstacle).filter((o) => o !== null),
    anchors: parseAnchors(v.anchors),
    northYawDeg: num(v.north_yaw_deg) ?? 0,
  };
}

function parseCoverage(v: unknown): Coverage | null {
  if (!isRecord(v)) return null;
  const cells: CoverageCell[] = [];
  for (const item of list(v.cells)) {
    if (!Array.isArray(item)) continue;
    const x = num(item[0]);
    const y = num(item[1]);
    if (x !== null && y !== null) cells.push([x, y, num(item[2]) ?? 0]);
  }
  return { cell: num(v.cell) ?? 0.25, cells, shareSeen: num(v.share_seen) ?? 0 };
}

function parseThing(v: unknown, index: number): Thing | null {
  const t = record(v);
  const p = vec3(t.p);
  const x = num(t.x) ?? p?.[0] ?? null;
  const y = num(t.y) ?? p?.[1] ?? null;
  if (x === null || y === null) return null;
  return { id: text(t.id) ?? String(index), label: text(t.label) ?? "thing", x, y, sure: t.sure !== false };
}

function parseAgent(v: unknown): AgentInfo {
  const a = record(v);
  return { model: text(a.model) ?? "", busy: a.busy === true, online: a.online !== false };
}

function describeStep(v: unknown): string {
  if (typeof v === "string") return v;
  const step = record(v);
  const words = firstText(step, ["text", "summary", "description"]);
  if (words) return words;
  const name = firstText(step, ["tool", "name", "action"]);
  const args = brief(step.args ?? step.arguments);
  if (name) return args ? `${name} ${args}` : name;
  return brief(v) ?? "";
}

export function parsePlan(v: unknown): Plan | null {
  if (!isRecord(v)) return null;
  const id = text(v.id) ?? text(v.plan_id);
  if (id === null) return null;
  const kind = text(v.kind);
  const region = text(v.region);
  const fallback = kind ? `Start a ${kind} mission${region ? ` over ${region}` : ""}.` : "Fly the plan below.";
  const steps = list(v.steps ?? v.tasks ?? v.actions ?? v.tool_calls).map(describeStep).filter(Boolean);
  return { id, summary: firstText(v, ["summary", "text", "description", "title"]) ?? fallback, steps };
}

export function parseState(raw: unknown): ZwarmState {
  if (!isRecord(raw)) throw new Error("the state response isn't a JSON object");
  const mission = parseMission(raw.mission);
  const missionRecord = record(raw.mission);
  return {
    simulated: raw.simulated === true,
    estopped: raw.estopped === true,
    flightLimitS: num(raw.flight_limit_s) ?? num(missionRecord.flight_limit_s) ?? DEFAULT_FLIGHT_LIMIT_S,
    mission,
    drones: list(raw.drones).map((d) => parseDrone(d, record(missionRecord.flight_s))).filter((d) => d !== null),
    room: parseRoom(raw.room),
    coverage: parseCoverage(raw.coverage),
    things: list(raw.things).map(parseThing).filter((t) => t !== null),
    agent: parseAgent(raw.agent),
    pendingPlan: parsePlan(raw.pending_plan),
  };
}

const isEventKind = (v: unknown): v is EventKind => EVENT_KINDS.some((kind) => kind === v);

function toolDetail(data: Json): { detail: string | null; refused: boolean } {
  const refused = data.ok === false || data.refused === true || typeof data.refused === "string";
  const detail = brief(data.result ?? data.refused ?? data.error ?? data.detail);
  return { detail, refused };
}

function parseEvent(v: unknown): LogEvent | null {
  const e = record(v);
  const n = num(e.n);
  if (n === null) return null;
  const data = record(e.data);
  return {
    n,
    t: num(e.t) ?? 0,
    kind: isEventKind(e.kind) ? e.kind : "report",
    text: text(e.text) ?? "",
    ...toolDetail(data),
  };
}

export function parseEvents(raw: unknown): LogEvent[] {
  const items = Array.isArray(raw) ? raw : list(record(raw).events);
  return items.map(parseEvent).filter((e) => e !== null);
}

function parseToolCall(v: unknown): ToolCall {
  if (typeof v === "string") return { text: v, detail: null, refused: false };
  const call = record(v);
  const name = firstText(call, ["text", "name", "tool"]) ?? "tool";
  const args = brief(call.args ?? call.arguments);
  return { text: args && !call.text ? `${name} ${args}` : name, ...toolDetail(call) };
}

export function parseChat(raw: unknown): ChatReply {
  const r = record(raw);
  return { reply: text(r.reply) ?? "", toolCalls: list(r.tool_calls).map(parseToolCall) };
}

const isFindingStatus = (v: unknown): v is FindingStatus => v === "pending" || v === "confirmed" || v === "dismissed";

function parseFinding(v: unknown): Finding | null {
  const f = record(v);
  const id = text(f.id);
  if (id === null) return null;
  return {
    id,
    t: num(f.t) ?? 0,
    drone: text(f.drone) ?? "",
    x: num(f.x),
    y: num(f.y),
    z: num(f.z),
    summary: text(f.summary) ?? "No description.",
    severity: text(f.severity) ?? "unrated",
    confidence: num(f.confidence),
    status: isFindingStatus(f.status) ? f.status : "pending",
    target: text(f.target),
  };
}

export function parseFindings(raw: unknown): Finding[] {
  const items = Array.isArray(raw) ? raw : list(record(raw).findings);
  return items.map(parseFinding).filter((f) => f !== null);
}

export function parseMapStatus(raw: unknown): MapStatus {
  const m = record(raw);
  const byDrone: Record<string, number> = {};
  for (const [drone, count] of Object.entries(record(m.by_drone))) byDrone[drone] = num(count) ?? 0;
  return {
    name: text(m.name) ?? "",
    state: text(m.state) ?? "idle",
    photos: num(m.photos) ?? 0,
    byDrone,
    registered: num(m.registered) ?? 0,
    error: text(m.error),
  };
}

export function parsePoints(raw: unknown): Cloud {
  const items = Array.isArray(raw) ? raw : list(record(raw).points);
  const xyz = new Float32Array(items.length * 3);
  const rgb = new Uint8ClampedArray(items.length * 3);
  let count = 0;
  let brightest = 0;
  for (const item of items) {
    if (!Array.isArray(item)) continue;
    const x = num(item[0]);
    const y = num(item[1]);
    const z = num(item[2]);
    if (x === null || y === null || z === null) continue;
    xyz.set([x, y, z], count * 3);
    const r = num(item[3]) ?? 200;
    const g = num(item[4]) ?? 200;
    const b = num(item[5]) ?? 200;
    brightest = Math.max(brightest, r, g, b);
    rgb.set([r, g, b], count * 3);
    count += 1;
  }
  // Some writers store colours as 0..1, and Uint8ClampedArray has already rounded those to 0 or 1.
  if (brightest > 0 && brightest <= 1) {
    for (let i = 0; i < count * 3; i += 1) rgb[i] = rgb[i] * 255;
  }
  return { count, xyz, rgb };
}
